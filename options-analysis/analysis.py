import glob
import os
import re

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Data Loading & Cleaning ─────────────────────────────────────────────────

def load_and_clean(file_paths):
    dfs = []
    for fp in file_paths:
        try:
            # Fidelity exports have 2 blank rows before the header row
            df = pd.read_csv(fp, skiprows=2)
            if 'Run Date' not in df.columns:
                df = pd.read_csv(fp)
            # A split fill (same contract, price and day) appears as identical
            # rows within one export, and those are real trades. Number repeats
            # within each file so only rows repeated ACROSS overlapping exports
            # are dropped below — a plain drop_duplicates() would delete fills.
            df['_dup_seq'] = df.groupby(list(df.columns), dropna=False).cumcount()
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {fp}: {e}")
    raw = pd.concat(dfs, ignore_index=True)
    raw['Run Date'] = pd.to_datetime(raw['Run Date'], format='%m/%d/%Y', errors='coerce')
    df = raw.dropna(subset=['Run Date']).copy()
    df = df[(df['Run Date'] >= '2025-01-01') & (df['Run Date'] <= '2026-12-31')]
    return df.drop_duplicates().drop(columns='_dup_seq')

def parse_action_type(action):
    action = str(action).upper()
    if 'EXPIRED' in action:
        return 'EXPIRED'
    elif 'ASSIGNED' in action:
        return 'ASSIGNED'
    elif 'SOLD OPENING' in action:
        return 'SELL_OPEN'
    elif 'BOUGHT OPENING' in action:
        return 'BUY_OPEN'
    elif 'SOLD CLOSING' in action:
        return 'SELL_CLOSE'
    elif 'BOUGHT CLOSING' in action:
        return 'BUY_CLOSE'
    return 'OTHER'

def parse_option_type(action):
    action = str(action).upper()
    if ' CALL ' in action:
        return 'CALL'
    elif ' PUT ' in action:
        return 'PUT'
    return 'UNKNOWN'

def parse_underlying(symbol):
    s = str(symbol).strip().lstrip('-')
    m = re.match(r'([A-Z]+)', s)
    return m.group(1) if m else s

def enrich(df):
    df['action_type'] = df['Action'].apply(parse_action_type)
    df['option_type'] = df['Action'].apply(parse_option_type)
    df['underlying']  = df['Symbol'].apply(parse_underlying)
    df['weekday']     = df['Run Date'].dt.day_name()
    for col in ['Amount', 'Price', 'Quantity', 'Commission', 'Fees']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['month']     = df['Run Date'].dt.to_period('M')
    df['month_str'] = df['month'].astype(str)
    return df

# ── Realized P&L (FIFO) ─────────────────────────────────────────────────────
#
# Summing Amount by transaction date gives net CASH FLOW, not P&L: opening a
# credit spread brings premium in before its outcome is known, so a month with
# many new positions looks profitable while its losses are still unresolved.
# Realized P&L matches each close against its opening lot (FIFO, per account
# and contract) and books the result in the month the position closed.

OPEN_ACTIONS  = {'SELL_OPEN', 'BUY_OPEN'}
CLOSE_ACTIONS = {'BUY_CLOSE', 'SELL_CLOSE', 'EXPIRED', 'ASSIGNED'}


def option_legs(df):
    """Option rows only (no stock trades, dividends or interest)."""
    return df[df['option_type'].isin(['PUT', 'CALL']) &
              df['action_type'].isin(OPEN_ACTIONS | CLOSE_ACTIONS)].copy()


def realize_fifo(df):
    """
    Match closes to opening lots FIFO per (Account, Symbol).

    Returns (realized, open_premium, pre_window):
      realized     — one row per matched chunk: open/close dates, P&L, close month.
                     Expired/assigned legs close at $0 (option leg only — the
                     stock side of an assignment is not in this P&L).
      open_premium — premium from lots still open at the end of the data.
      pre_window   — closes with no opening lot in the data (opened before the
                     first export); excluded from realized P&L.
    Every option cash flow lands in exactly one bucket, so
    realized + open_premium + pre_window == option cash flow.
    """
    legs = option_legs(df)
    legs = pd.DataFrame({
        'date': legs['Run Date'], 'account': legs['Account'], 'symbol': legs['Symbol'],
        'underlying': legs['underlying'], 'option_type': legs['option_type'],
        'qty': legs['Quantity'].abs(), 'amount': legs['Amount'],
        'is_close': legs['action_type'].isin(CLOSE_ACTIONS),
    })
    # same-day opens before closes, so a same-day round trip matches
    legs = legs.sort_values(['date', 'is_close'], kind='stable')

    lots, rows, pre_window = {}, [], 0.0
    for r in legs.itertuples(index=False):
        qty = r.qty
        if qty == 0:
            continue
        per = r.amount / qty
        key = (r.account, r.symbol)
        if not r.is_close:
            lots.setdefault(key, []).append([qty, per, r.date])
            continue
        queue = lots.get(key, [])
        while qty > 1e-9 and queue:
            lot = queue[0]
            m = min(qty, lot[0])
            rows.append({
                'Account': r.account, 'Symbol': r.symbol,
                'underlying': r.underlying, 'option_type': r.option_type,
                'open_date': lot[2], 'close_date': r.date,
                'contracts': m, 'pnl': m * (lot[1] + per),
            })
            lot[0] -= m
            qty -= m
            if lot[0] <= 1e-9:
                queue.pop(0)
        if qty > 1e-9:
            pre_window += qty * per

    realized = pd.DataFrame(rows, columns=['Account', 'Symbol', 'underlying', 'option_type',
                                           'open_date', 'close_date', 'contracts', 'pnl'])
    realized['month_str'] = pd.to_datetime(realized['close_date']).dt.to_period('M').astype(str)
    realized['quarter']   = pd.to_datetime(realized['close_date']).dt.to_period('Q').astype(str)
    realized['open_weekday'] = pd.to_datetime(realized['open_date']).dt.day_name()
    open_premium = sum(q * per for queue in lots.values() for q, per, _ in queue)
    return realized, open_premium, pre_window


# ── Analysis ────────────────────────────────────────────────────────────────

def print_summary(df, realized, open_premium, pre_window):
    legs = option_legs(df)
    cash = legs['Amount'].sum()
    rpnl = realized['pnl'].sum()
    print('=' * 50)
    print('SUMMARY')
    print('=' * 50)
    print(f"Option legs        : {len(legs):,}")
    print(f"Date range         : {legs['Run Date'].min().date()} to {legs['Run Date'].max().date()}")
    print(f"Realized P&L (FIFO): ${rpnl:,.2f}")
    print(f"Net cash flow      : ${cash:,.2f}")
    print(f"  = realized + open premium ${open_premium:,.2f} + pre-window closes ${pre_window:,.2f}"
          f"  (check: {'OK' if abs(rpnl + open_premium + pre_window - cash) < 0.01 else 'MISMATCH'})")
    print(f"Fees paid          : ${(legs['Commission'] + legs['Fees']).sum():,.2f} (already in both figures)")

    monthly = pd.DataFrame({
        'realized_pnl': realized.groupby('month_str')['pnl'].sum(),
        'net_cash_flow': legs.groupby('month_str')['Amount'].sum(),
    }).fillna(0).sort_index()
    win = (monthly['realized_pnl'] > 0).sum()
    cash_pos = (monthly['net_cash_flow'] > 0).sum()
    print(f"Monthly win rate   : {win / len(monthly):.0%} ({win}/{len(monthly)} months, realized P&L)")
    print(f"  cash-flow basis  : {cash_pos}/{len(monthly)} months positive — not a P&L measure")
    print()
    print('── Monthly: realized P&L vs net cash flow ──')
    print(monthly.round(2).to_string())
    print()
    print('── Realized P&L by Account ──')
    print(realized.groupby('Account')['pnl'].sum().sort_values(ascending=False).round(2).to_string())
    print()
    print('── Realized P&L by Underlying (worst to best) ──')
    print(realized.groupby('underlying')['pnl'].sum().sort_values().round(2).to_string())
    print()
    print('── Realized P&L: PUT vs CALL (option leg only) ──')
    print(realized.groupby('option_type')['pnl'].sum().round(2).to_string())

# ── Visualizations ──────────────────────────────────────────────────────────

def _month_label(m):
    return m[-2:] + "'" + m[2:4]

def make_charts(df, realized, out_dir='charts'):
    """P&L charts use realized P&L (FIFO, by close date); activity charts use opening trades."""
    os.makedirs(out_dir, exist_ok=True)

    monthly = realized.groupby('month_str')['pnl'].sum().reset_index()
    monthly.columns = ['month', 'pnl']
    monthly['cum_pnl'] = monthly['pnl'].cumsum()
    monthly['label'] = monthly['month'].apply(_month_label)

    total_pnl = realized['pnl'].sum()
    date_start = df['Run Date'].min().strftime('%b %Y')
    date_end   = df['Run Date'].max().strftime('%b %Y')
    cum_final  = monthly['cum_pnl'].iloc[-1]

    opt_pnl_raw = realized.groupby('option_type')['pnl'].sum()
    put_pnl  = opt_pnl_raw.get('PUT', 0)
    call_pnl = opt_pnl_raw.get('CALL', 0)
    gross    = abs(put_pnl) + abs(call_pnl)
    put_pct  = int(abs(put_pnl) / gross * 100) if gross else 0

    und_pnl_s = realized.groupby('underlying')['pnl'].sum().sort_values()
    top_winner = und_pnl_s.index[-1]
    top_loser  = und_pnl_s.index[0]

    # Chart 1: Monthly P&L + Cumulative
    bar_colors = ['#e74c3c' if v < 0 else '#00d4a8' for v in monthly['pnl']]
    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.6, 0.4], vertical_spacing=0.08,
                         subplot_titles=('Monthly P&L', 'Cumulative P&L'))
    fig1.add_trace(go.Bar(
        x=monthly['label'], y=monthly['pnl'],
        marker_color=bar_colors,
        text=[f"${v/1000:.1f}k" for v in monthly['pnl']],
        textposition='outside', textfont=dict(size=10),
        width=0.55, showlegend=False
    ), row=1, col=1)
    fig1.add_trace(go.Scatter(
        x=monthly['label'], y=monthly['cum_pnl'],
        mode='lines+markers',
        line=dict(color='#f39c12', width=3), marker=dict(size=8),
        fill='tozeroy', fillcolor='rgba(243,156,18,0.15)', showlegend=False
    ), row=2, col=1)
    fig1.update_layout(title={'text': f'Cumulative Realized P&L Reached ${cum_final/1000:.1f}k<br>'
                                       f"<span style='font-size:16px;font-weight:normal;'>All Accounts | {date_start} – {date_end} | by close month</span>"})
    fig1.update_yaxes(title_text='P&L ($)', tickformat='$,.0f', row=1, col=1)
    fig1.update_yaxes(title_text='Cumul. ($)', tickformat='$,.0f', row=2, col=1)
    fig1.update_xaxes(title_text='Month', row=2, col=1)
    fig1.write_image(f'{out_dir}/chart1_monthly_pnl.png')

    # Chart 2: Ticker P&L horizontal bar
    top_combo = pd.concat([und_pnl_s.head(4), und_pnl_s.tail(10)])
    fig2 = go.Figure(go.Bar(
        x=top_combo.values,
        y=[f'  {t}' for t in top_combo.index],
        orientation='h',
        marker_color=['#e74c3c' if v < 0 else '#00d4a8' for v in top_combo.values],
        text=[f'${v:,.0f}' for v in top_combo.values],
        textposition='auto', textfont=dict(size=11)
    ))
    fig2.update_layout(
        title={'text': f'{top_winner} Leads Gains; {top_loser} Biggest Drag<br>'
                       "<span style='font-size:16px;font-weight:normal;'>Realized P&L by Ticker | Top 10 + 4 Losers</span>"},
        xaxis=dict(title_text='Realized P&L ($)', tickformat='$,.0f'),
        height=700, margin=dict(l=100, r=80, t=120, b=60)
    )
    fig2.write_image(f'{out_dir}/chart2_ticker_pnl.png')

    # Chart 3: PUT vs CALL donut
    opt_pnl = opt_pnl_raw.reset_index()
    opt_pnl.columns = ['option_type', 'pnl']
    fig3 = go.Figure(go.Pie(
        labels=opt_pnl['option_type'], values=opt_pnl['pnl'].abs(),
        hole=0.4, marker_colors=['#00d4a8', '#e74c3c'],
        textinfo='label+percent', textfont_size=14, pull=[0.03, 0.03]
    ))
    fig3.update_layout(
        title={'text': f'PUT Spreads Drive {put_pct}% of Gross Realized P&L<br>'
                       f"<span style='font-size:16px;font-weight:normal;'>PUT: ${put_pnl:+,.0f} | CALL: ${call_pnl:+,.0f}</span>"},
        legend=dict(orientation='v', x=1.0)
    )
    fig3.write_image(f'{out_dir}/chart3_put_vs_call.png')

    # Chart 4: Monthly trade count
    tc = df[df['action_type'] == 'SELL_OPEN'].groupby('month_str').size().reset_index()
    tc.columns = ['month', 'count']
    fig4 = go.Figure(go.Bar(
        x=tc['month'].apply(_month_label), y=tc['count'],
        marker_color='#3498db',
        text=tc['count'], textposition='outside', width=0.6
    ))
    fig4.update_layout(
        title={'text': f'New Positions (SELL_OPEN) per Month<br>'
                       f"<span style='font-size:16px;font-weight:normal;'>{date_start} – {date_end} | All Accounts</span>"},
        xaxis=dict(title_text='Month'),
        yaxis=dict(title_text='# of Trades')
    )
    fig4.write_image(f'{out_dir}/chart4_trade_count.png')

    # Chart 5: Account P&L
    acct = realized.groupby('Account')['pnl'].sum().sort_values(ascending=False).reset_index()
    name_map = {'Individual': 'Account A', 'Rollover IRA': 'Account B',
                'ROTH IRA': 'Account C', 'Health Savings Account': 'Account D'}
    acct['short'] = acct['Account'].map(name_map).fillna(acct['Account'])
    fig5 = go.Figure(go.Bar(
        x=acct['short'], y=acct['pnl'],
        marker_color=['#3498db', '#9b59b6', '#e67e22', '#1abc9c'],
        text=[f'${v:,.0f}' for v in acct['pnl']],
        textposition='inside', textfont=dict(size=13, color='white'), width=0.45
    ))
    fig5.update_layout(
        title={'text': f'Realized P&L by Account | Total ${total_pnl:,.0f}<br>'
                       f"<span style='font-size:16px;font-weight:normal;'>{date_start} – {date_end}</span>"},
        xaxis=dict(title_text='Account'),
        yaxis=dict(title_text='Realized P&L ($)', tickformat='$,.0f'),
        margin=dict(l=80, r=40, t=120, b=60)
    )
    fig5.write_image(f'{out_dir}/chart5_account_pnl.png')

    # Chart 6: Weekday P&L
    wday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    # Attributed to the day the position was OPENED — that's the decision being evaluated.
    wday = realized.groupby('open_weekday')['pnl'].sum().reindex(wday_order).fillna(0)
    best_day = wday.idxmax() if not wday.empty else 'n/a'
    wday = wday.reset_index()
    wday.columns = ['weekday', 'pnl']
    fig6 = go.Figure(go.Bar(
        x=wday['weekday'], y=wday['pnl'],
        marker_color=['#e74c3c' if v < 0 else '#00d4a8' for v in wday['pnl']],
        text=[f'${v:,.0f}' for v in wday['pnl']], textposition='outside', width=0.5
    ))
    fig6.update_layout(
        title={'text': f'{best_day} Entries Realized the Most P&L<br>'
                       "<span style='font-size:16px;font-weight:normal;'>Realized P&L by Entry Day of Week | All Accounts</span>"},
        xaxis=dict(title_text='Entry Day of Week'),
        yaxis=dict(title_text='Realized P&L ($)', tickformat='$,.0f')
    )
    fig6.write_image(f'{out_dir}/chart6_weekday_pnl.png')

    # Chart 7: Ticker frequency
    freq = df[df['action_type'] == 'SELL_OPEN'].groupby('underlying').size().sort_values(ascending=False).head(12)
    fig7 = go.Figure(go.Bar(
        x=freq.values,
        y=[f'  {t}' for t in freq.index],
        orientation='h',
        marker_color='#9b59b6',
        text=freq.values,
        textposition='inside', textfont=dict(size=12, color='white')
    ))
    fig7.update_layout(
        title={'text': f'{freq.index[0]} Leads with {freq.iloc[0]} Opening Trades<br>'
                       "<span style='font-size:16px;font-weight:normal;'>New Positions (SELL_OPEN) by Ticker | Top 12</span>"},
        xaxis=dict(title_text='Trades'),
        height=500, margin=dict(l=90, r=60, t=120, b=60)
    )
    fig7.write_image(f'{out_dir}/chart7_ticker_frequency.png')

    # Chart 8: Quarterly P&L trajectory
    qtr = realized.groupby('quarter')['pnl'].sum().reset_index()
    qtr.columns = ['quarter', 'pnl']
    qtr['cum_pnl'] = qtr['pnl'].cumsum()
    fig8 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.6, 0.4], vertical_spacing=0.08,
                         subplot_titles=('Quarterly P&L', 'Cumulative P&L'))
    fig8.add_trace(go.Bar(
        x=qtr['quarter'], y=qtr['pnl'],
        marker_color=['#e74c3c' if v < 0 else '#00d4a8' for v in qtr['pnl']],
        text=[f"${v/1000:.1f}k" for v in qtr['pnl']],
        textposition='outside', width=0.5, showlegend=False
    ), row=1, col=1)
    fig8.add_trace(go.Scatter(
        x=qtr['quarter'], y=qtr['cum_pnl'],
        mode='lines+markers',
        line=dict(color='#f39c12', width=3), marker=dict(size=8),
        fill='tozeroy', fillcolor='rgba(243,156,18,0.15)', showlegend=False
    ), row=2, col=1)
    fig8.update_layout(title={'text': 'Quarterly Realized P&L Trajectory<br>'
                                      "<span style='font-size:16px;font-weight:normal;'>By close quarter | All Accounts</span>"})
    fig8.update_yaxes(title_text='P&L ($)', tickformat='$,.0f', row=1, col=1)
    fig8.update_yaxes(title_text='Cumul. ($)', tickformat='$,.0f', row=2, col=1)
    fig8.write_image(f'{out_dir}/chart8_quarterly_pnl.png')

    # Chart 9: Ticker efficiency — P&L per trade vs trade count
    sell_open = df[df['action_type'] == 'SELL_OPEN']
    t_count = sell_open.groupby('underlying').size().reset_index(name='trade_count')
    t_pnl   = realized.groupby('underlying')['pnl'].sum().reset_index(name='total_pnl')
    t_stats = t_count.merge(t_pnl, on='underlying')
    t_stats = t_stats[t_stats['trade_count'] >= 3].copy()
    t_stats['pnl_per_trade'] = t_stats['total_pnl'] / t_stats['trade_count']
    fig9 = go.Figure(go.Scatter(
        x=t_stats['trade_count'],
        y=t_stats['pnl_per_trade'],
        mode='markers+text',
        text=t_stats['underlying'],
        textposition='top center',
        textfont=dict(size=11),
        marker=dict(
            size=[max(10, min(45, abs(v) / 80)) for v in t_stats['total_pnl']],
            color=['#e74c3c' if v < 0 else '#00d4a8' for v in t_stats['total_pnl']],
            opacity=0.85,
            line=dict(width=1, color='white'),
        )
    ))
    fig9.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
    fig9.update_layout(
        title={'text': 'Ticker Efficiency: P&L per Trade vs Frequency<br>'
                       "<span style='font-size:16px;font-weight:normal;'>Bubble size = |Total P&L| · green = profit · red = loss</span>"},
        xaxis=dict(title_text='Number of Trades (SELL_OPEN)'),
        yaxis=dict(title_text='P&L per Trade ($)', tickformat='$,.0f'),
        height=620, margin=dict(l=80, r=60, t=120, b=60)
    )
    fig9.write_image(f'{out_dir}/chart9_ticker_efficiency.png')

    print(f'✅  9 charts saved to ./{out_dir}/')

# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    files = glob.glob('data/*.csv')
    files = [f for f in files if 'options_cleaned' not in f]
    if not files:
        print('No CSV files found in data/. Add Fidelity export files and retry.')
    else:
        df = load_and_clean(files)
        df = enrich(df)
        realized, open_premium, pre_window = realize_fifo(df)
        print_summary(df, realized, open_premium, pre_window)
        df.to_csv('data/options_cleaned.csv', index=False)
        realized.to_csv('data/realized_pnl.csv', index=False)
        print('\nSaved cleaned data → data/options_cleaned.csv')
        print('Saved realized P&L → data/realized_pnl.csv')
        make_charts(df, realized)

import pandas as pd
import re
import glob
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Data Loading & Cleaning ─────────────────────────────────────────────────

def load_and_clean(file_paths):
    dfs = []
    for fp in file_paths:
        try:
            df = pd.read_csv(fp)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {fp}: {e}")
    raw = pd.concat(dfs, ignore_index=True)
    raw['Run Date'] = pd.to_datetime(raw['Run Date'], format='%m/%d/%Y', errors='coerce')
    df = raw.dropna(subset=['Run Date']).copy()
    df = df[(df['Run Date'] >= '2025-01-01') & (df['Run Date'] <= '2026-03-31')]
    return df.drop_duplicates()

def parse_action_type(action):
    action = str(action).upper()
    if 'EXPIRED' in action:          return 'EXPIRED'
    elif 'ASSIGNED' in action:       return 'ASSIGNED'
    elif 'SOLD OPENING' in action:   return 'SELL_OPEN'
    elif 'BOUGHT OPENING' in action: return 'BUY_OPEN'
    elif 'SOLD CLOSING' in action:   return 'SELL_CLOSE'
    elif 'BOUGHT CLOSING' in action: return 'BUY_CLOSE'
    return 'OTHER'

def parse_option_type(action):
    action = str(action).upper()
    if ' CALL ' in action: return 'CALL'
    elif ' PUT ' in action: return 'PUT'
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

# ── Analysis ────────────────────────────────────────────────────────────────

def print_summary(df):
    print('=' * 50)
    print('SUMMARY')
    print('=' * 50)
    print(f"Total transactions : {len(df):,}")
    print(f"Date range         : {df['Run Date'].min().date()} to {df['Run Date'].max().date()}")
    print(f"Total Net P&L      : ${df['Amount'].sum():,.2f}")
    print(f"Total Fees Paid    : ${(df['Commission'] + df['Fees']).sum():,.2f}")
    monthly = df.groupby('month_str')['Amount'].sum()
    win_rate = (monthly > 0).sum() / len(monthly)
    print(f"Monthly Win Rate   : {win_rate:.0%} ({(monthly > 0).sum()}/{len(monthly)} months)")
    print()
    print('── Monthly P&L ──')
    print(monthly.to_string())
    print()
    print('── P&L by Account ──')
    print(df.groupby('Account')['Amount'].sum().sort_values(ascending=False).to_string())
    print()
    print('── P&L by Underlying (worst to best) ──')
    print(df.groupby('underlying')['Amount'].sum().sort_values().to_string())
    print()
    print('── PUT vs CALL ──')
    print(df.groupby('option_type')['Amount'].sum().to_string())

# ── Visualizations ──────────────────────────────────────────────────────────

def _month_label(m):
    return m[-2:] + "'" + m[2:4]

def make_charts(df, out_dir='charts'):
    os.makedirs(out_dir, exist_ok=True)

    monthly = df.groupby('month_str')['Amount'].sum().reset_index()
    monthly.columns = ['month', 'pnl']
    monthly['cum_pnl'] = monthly['pnl'].cumsum()
    monthly['label'] = monthly['month'].apply(_month_label)

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
    fig1.update_layout(title={'text': 'Monthly P&L Surged in H2; Cumulative Reached $28.7k<br>'
                                       "<span style='font-size:16px;font-weight:normal;'>All Accounts | Jan 2025 – Dec 2025</span>"})
    fig1.update_yaxes(title_text='P&L ($)', tickformat='$,.0f', row=1, col=1)
    fig1.update_yaxes(title_text='Cumul. ($)', tickformat='$,.0f', row=2, col=1)
    fig1.update_xaxes(title_text='Month', row=2, col=1)
    fig1.write_image(f'{out_dir}/chart1_monthly_pnl.png')

    # Chart 2: Ticker P&L horizontal bar
    und_pnl = df.groupby('underlying')['Amount'].sum().sort_values()
    top_combo = pd.concat([und_pnl.head(4), und_pnl.tail(10)])
    fig2 = go.Figure(go.Bar(
        x=top_combo.values,
        y=[f'  {t}' for t in top_combo.index],
        orientation='h',
        marker_color=['#e74c3c' if v < 0 else '#00d4a8' for v in top_combo.values],
        text=[f'${v:,.0f}' for v in top_combo.values],
        textposition='auto', textfont=dict(size=11)
    ))
    fig2.update_layout(
        title={'text': 'NVDA & HIMS Lead Gains; PLTR Biggest Drag<br>'
                       "<span style='font-size:16px;font-weight:normal;'>Net P&L by Ticker | Top 10 + 4 Losers</span>"},
        xaxis=dict(title_text='Net P&L ($)', tickformat='$,.0f'),
        height=700, margin=dict(l=100, r=80, t=120, b=60)
    )
    fig2.write_image(f'{out_dir}/chart2_ticker_pnl.png')

    # Chart 3: PUT vs CALL donut
    opt_pnl = df[df['option_type'].isin(['PUT', 'CALL'])].groupby('option_type')['Amount'].sum().reset_index()
    fig3 = go.Figure(go.Pie(
        labels=opt_pnl['option_type'], values=opt_pnl['Amount'].abs(),
        hole=0.4, marker_colors=['#00d4a8', '#e74c3c'],
        textinfo='label+percent', textfont_size=14, pull=[0.03, 0.03]
    ))
    fig3.update_layout(
        title={'text': 'PUT Spreads Drive 92% of Gross P&L<br>'
                       "<span style='font-size:16px;font-weight:normal;'>PUT: +$31,496 | CALL: -$2,815</span>"},
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
        title={'text': 'Trade Frequency Grew ~27x from Jan to Oct 2025<br>'
                       "<span style='font-size:16px;font-weight:normal;'>SELL_OPEN Transactions per Month</span>"},
        xaxis=dict(title_text='Month'),
        yaxis=dict(title_text='# of Trades')
    )
    fig4.write_image(f'{out_dir}/chart4_trade_count.png')

    # Chart 5: Account P&L
    acct = df.groupby('Account')['Amount'].sum().sort_values(ascending=False).reset_index()
    name_map = {'Individual': 'Individual', 'Rollover IRA': 'Rollover IRA',
                'ROTH IRA': 'Roth IRA', 'Health Savings Account': 'HSA'}
    acct['short'] = acct['Account'].map(name_map).fillna(acct['Account'])
    fig5 = go.Figure(go.Bar(
        x=acct['short'], y=acct['Amount'],
        marker_color=['#3498db', '#9b59b6', '#e67e22', '#1abc9c'],
        text=[f'${v:,.0f}' for v in acct['Amount']],
        textposition='inside', textfont=dict(size=13, color='white'), width=0.45
    ))
    fig5.update_layout(
        title={'text': 'Individual Account Earned 56% of Total P&L<br>'
                       "<span style='font-size:16px;font-weight:normal;'>Net P&L by Account | Total $28,681</span>"},
        xaxis=dict(title_text='Account'),
        yaxis=dict(title_text='Net P&L ($)', tickformat='$,.0f'),
        margin=dict(l=80, r=40, t=120, b=60)
    )
    fig5.write_image(f'{out_dir}/chart5_account_pnl.png')

    # Chart 6: Weekday P&L
    wday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    wday = df.groupby('weekday')['Amount'].sum().reindex(wday_order).reset_index()
    fig6 = go.Figure(go.Bar(
        x=wday['weekday'], y=wday['Amount'],
        marker_color=['#e74c3c' if v < 0 else '#00d4a8' for v in wday['Amount']],
        text=[f'${v:,.0f}' for v in wday['Amount']], textposition='outside', width=0.5
    ))
    fig6.update_layout(
        title={'text': 'Wednesday & Thursday Are Your Best Trading Days<br>'
                       "<span style='font-size:16px;font-weight:normal;'>Net P&L by Day of Week | All Accounts</span>"},
        xaxis=dict(title_text='Day of Week'),
        yaxis=dict(title_text='Net P&L ($)', tickformat='$,.0f')
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
        title={'text': f'NVDA Leads with {freq.iloc[0]} Opening Trades in 2025<br>'
                       "<span style='font-size:16px;font-weight:normal;'>New Positions (SELL_OPEN) by Ticker | Top 12</span>"},
        xaxis=dict(title_text='Trades'),
        height=500, margin=dict(l=90, r=60, t=120, b=60)
    )
    fig7.write_image(f'{out_dir}/chart7_ticker_frequency.png')

    print(f'✅  7 charts saved to ./{out_dir}/')

# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    files = glob.glob('data/Accounts_History*.csv')
    if not files:
        print('No CSV files found in data/. Add Fidelity export files and retry.')
    else:
        df = load_and_clean(files)
        df = enrich(df)
        print_summary(df)
        df.to_csv('data/options_cleaned.csv', index=False)
        print('\nSaved cleaned data → data/options_cleaned.csv')
        make_charts(df)

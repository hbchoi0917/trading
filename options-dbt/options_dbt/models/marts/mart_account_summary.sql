{{
    config(
        materialized='table',
        description='Account-level lifetime performance summary. Validates against insights_report.md account table.'
    )
}}

with closed_spreads as (
    select * from {{ ref('int_spread_trades') }}
    where not is_open
),

monthly_pnl as (
    select * from {{ ref('mart_monthly_pnl') }}
),

account_stats as (
    select
        account_name,

        count(*)                                                        as total_trades,
        sum(case when trade_result = 'WIN'  then 1 else 0 end)         as total_wins,
        sum(case when trade_result = 'LOSS' then 1 else 0 end)         as total_losses,

        round(sum(realized_pnl), 2)                                     as lifetime_pnl,
        round(sum(total_fees), 2)                                       as lifetime_fees,
        round(sum(realized_pnl) - sum(total_fees), 2)                   as lifetime_pnl_net_fees,

        round(avg(realized_pnl), 2)                                     as avg_pnl_per_trade,
        round(max(realized_pnl), 2)                                     as best_single_trade,
        round(min(realized_pnl), 2)                                     as worst_single_trade,

        count(distinct ticker)                                          as unique_tickers_traded,
        count(distinct date_trunc('month', close_date))                 as active_months,

        min(open_date)                                                  as first_trade_date,
        max(close_date)                                                  as last_trade_date

    from closed_spreads
    group by 1
),

win_months as (
    select
        account_name,
        count(*)                                                        as total_months,
        sum(case when is_win_month then 1 else 0 end)                  as win_months
    from monthly_pnl
    group by 1
),

-- max monthly drawdown per account
monthly_drawdown as (
    select
        account_name,
        round(min(net_pnl), 2)                                         as worst_month_pnl,
        round(max(net_pnl), 2)                                         as best_month_pnl
    from monthly_pnl
    group by 1
)

select
    a.account_name,
    a.total_trades,
    a.total_wins,
    a.total_losses,
    round(a.total_wins::double / nullif(a.total_trades, 0) * 100, 1)   as trade_win_rate_pct,
    round(w.win_months::double / nullif(w.total_months, 0) * 100, 1)   as monthly_win_rate_pct,
    a.lifetime_pnl,
    a.lifetime_fees,
    a.lifetime_pnl_net_fees,
    a.avg_pnl_per_trade,
    a.best_single_trade,
    a.worst_single_trade,
    d.best_month_pnl,
    d.worst_month_pnl,
    a.unique_tickers_traded,
    a.active_months,
    w.total_months,
    w.win_months,
    a.first_trade_date,
    a.last_trade_date

from account_stats a
left join win_months w      using (account_name)
left join monthly_drawdown d using (account_name)
order by lifetime_pnl desc

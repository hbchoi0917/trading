{{
    config(
        materialized='table',
        description='Monthly P&L summary by account. Closed trades only. Matches figures in insights_report.md.'
    )
}}

with closed_spreads as (
    select * from {{ ref('int_spread_trades') }}
    where not is_open
),

monthly as (
    select
        date_trunc('month', close_date)             as month,
        extract('year' from close_date)             as year,
        extract('month' from close_date)            as month_num,
        account_name,

        count(*)                                    as trade_count,
        sum(case when trade_result = 'WIN'  then 1 else 0 end)  as wins,
        sum(case when trade_result = 'LOSS' then 1 else 0 end)  as losses,
        round(sum(realized_pnl), 2)                 as net_pnl,
        round(sum(total_fees), 2)                   as total_fees,
        round(sum(realized_pnl) - sum(total_fees), 2) as net_pnl_after_fees

    from closed_spreads
    group by 1,2,3,4
),

with_totals as (
    select
        month,
        year,
        month_num,
        account_name,
        trade_count,
        wins,
        losses,
        net_pnl,
        total_fees,
        net_pnl_after_fees,

        -- win rate
        case
            when trade_count > 0
            then round(wins::double / trade_count * 100, 1)
            else null
        end                                         as win_rate_pct,

        -- cumulative P&L per account (window function)
        sum(net_pnl) over (
            partition by account_name
            order by month
            rows between unbounded preceding and current row
        )                                           as cumulative_pnl,

        -- month-over-month change
        net_pnl - lag(net_pnl) over (
            partition by account_name
            order by month
        )                                           as pnl_mom_change,

        -- is win month
        case when net_pnl > 0 then true else false end  as is_win_month

    from monthly
)

select * from with_totals
order by month, account_name

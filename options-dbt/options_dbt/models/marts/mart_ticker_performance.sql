{{
    config(
        materialized='table',
        description='Ticker-level performance summary across all accounts and periods. Key mart for portfolio rotation decisions.'
    )
}}

with closed_spreads as (
    select * from {{ ref('int_spread_trades') }}
    where not is_open
),

ticker_stats as (
    select
        ticker,
        option_type,

        count(*)                                                        as total_trades,
        sum(case when trade_result = 'WIN'  then 1 else 0 end)         as wins,
        sum(case when trade_result = 'LOSS' then 1 else 0 end)         as losses,

        round(sum(realized_pnl), 2)                                     as total_pnl,
        round(avg(realized_pnl), 2)                                     as avg_pnl_per_trade,
        round(max(realized_pnl), 2)                                     as best_trade,
        round(min(realized_pnl), 2)                                     as worst_trade,
        round(sum(total_fees), 2)                                       as total_fees,

        round(avg(holding_days), 1)                                     as avg_holding_days,
        round(avg(dte_at_open), 1)                                      as avg_dte_at_open,

        min(open_date)                                                  as first_trade_date,
        max(close_date)                                                  as last_trade_date,

        count(distinct account_name)                                    as accounts_active,
        string_agg(distinct account_name, ', ' order by account_name)  as account_list

    from (
        select
            s.*,
            l.dte_at_trade as dte_at_open
        from closed_spreads s
        left join {{ ref('int_option_legs') }} l
            on  l.ticker       = s.ticker
            and l.account_name = s.account_name
            and l.expiry_date  = s.expiry_date
            and l.strike       = s.strike
            and l.is_opening   = true
    )
    group by 1, 2
),

with_rank as (
    select
        ticker,
        option_type,
        total_trades,
        wins,
        losses,
        total_pnl,
        avg_pnl_per_trade,
        best_trade,
        worst_trade,
        total_fees,
        avg_holding_days,
        avg_dte_at_open,
        first_trade_date,
        last_trade_date,
        accounts_active,
        account_list,

        case
            when total_trades > 0
            then round(wins::double / total_trades * 100, 1)
            else null
        end                                                             as win_rate_pct,

        -- efficiency: P&L per trade relative to trade count (frequency-adjusted)
        case
            when total_trades > 0
            then round(total_pnl / total_trades, 2)
            else null
        end                                                             as pnl_per_trade,

        -- rank by total P&L
        rank() over (order by total_pnl desc)                          as pnl_rank,

        -- flag tickers excluded from rotation (based on insights_report)
        case
            when ticker in ('MSFT', 'NFLX', 'IONQ', 'RGTI')
            then true else false
        end                                                             as is_excluded

    from ticker_stats
)

select * from with_rank
order by total_pnl desc

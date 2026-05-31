{{
    config(
        materialized='view',
        description='Realized spread-level P&L. Groups opening and closing legs by (account, ticker, expiry, option_type, strike) to compute net cash flow per trade cycle.'
    )
}}

/*
  Grain: one row per spread trade cycle.
  A cycle = all legs sharing the same account + ticker + expiry + option_type + strike.

  Design decision: we group by (account, ticker, expiry, option_type, strike) rather than
  attempting sequential leg matching. This handles rolls naturally — a CLOSING leg followed
  by an OPENING leg on the same key creates two separate cycle rows.

  P&L = sum(amount) across all legs for the cycle.
  Positive = net received (profitable short spread). Negative = net paid (loss or long side).
*/

with legs as (
    select * from {{ ref('int_option_legs') }}
),

-- aggregate all legs per natural key
spread_agg as (
    select
        account_name,
        ticker,
        expiry_date,
        option_type,
        strike,
        margin_type,

        min(trade_date)                                                 as open_date,
        max(trade_date)                                                 as close_date,
        datediff('day', min(trade_date), max(trade_date))              as holding_days,

        sum(case when is_opening then amount else 0 end)               as entry_credit,
        sum(case when is_closing then amount else 0 end)               as exit_amount,
        sum(amount)                                                     as realized_pnl,
        sum(total_cost)                                                 as total_fees,
        sum(amount) - sum(total_cost)                                   as realized_pnl_gross,

        count(*)                                                        as leg_count,
        max(contracts)                                                  as contracts,

        -- close reason heuristic
        case
            when bool_or(transaction_type = 'EXPIRED')  then 'EXPIRED_WORTHLESS'
            when bool_or(transaction_type = 'ASSIGNED') then 'ASSIGNED'
            when max(trade_date) > min(trade_date)      then 'CLOSED'
            else 'OPEN'
        end                                                             as close_reason,

        -- is still open if only opening legs exist
        case
            when bool_and(is_opening) then true
            else false
        end                                                             as is_open

    from legs
    group by 1,2,3,4,5,6
),

with_metrics as (
    select
        md5(concat_ws('|',
            account_name,
            coalesce(ticker, ''),
            cast(coalesce(expiry_date, '1900-01-01') as varchar),
            coalesce(option_type, ''),
            cast(coalesce(strike, 0) as varchar)
        ))                                                              as spread_id,
        account_name,
        ticker,
        expiry_date,
        option_type,
        strike,
        margin_type,
        open_date,
        close_date,
        holding_days,
        entry_credit,
        exit_amount,
        realized_pnl,
        total_fees,
        realized_pnl_gross,
        leg_count,
        contracts,
        close_reason,
        is_open,

        -- win/loss classification
        case
            when is_open                then 'OPEN'
            when realized_pnl > 0      then 'WIN'
            when realized_pnl = 0      then 'BREAKEVEN'
            else                            'LOSS'
        end                                                             as trade_result,

        -- max theoretical risk for a short put spread (spread_width × 100 × contracts)
        -- we don't have long strike here so use entry_credit as lower bound proxy
        abs(entry_credit)                                               as max_risk_proxy,

        -- P&L as % of entry credit collected
        case
            when abs(entry_credit) > 0
            then round(realized_pnl / abs(entry_credit) * 100, 1)
            else null
        end                                                             as pnl_pct_of_credit,

        date_trunc('month', open_date)                                  as open_month,
        date_trunc('quarter', open_date)                                as open_quarter,
        extract('year' from open_date)                                  as open_year

    from spread_agg
)

select * from with_metrics

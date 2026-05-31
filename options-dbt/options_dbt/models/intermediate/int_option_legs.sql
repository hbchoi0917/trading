{{
    config(
        materialized='view',
        description='Option legs only (excludes dividends). Enriched with DTE, notional value, and spread-role classification.'
    )
}}

with stg as (
    select * from {{ ref('stg_fidelity_transactions') }}
    where transaction_category = 'OPTION'
),

enriched as (
    select
        transaction_id,
        trade_date,
        settlement_date,
        account_name,
        transaction_type,
        direction,
        ticker,
        expiry_date,
        option_type,
        strike,
        symbol_raw,
        margin_type,
        contracts,
        price_per_share,
        premium_per_contract,
        commission,
        fees,
        total_cost,
        amount,
        is_opening,
        is_closing,

        -- days to expiry at time of trade
        case
            when expiry_date is not null
            then datediff('day', trade_date, expiry_date)
            else null
        end                                                         as dte_at_trade,

        -- notional exposure (max risk for long put, collateral for short put)
        strike * 100 * contracts                                    as notional_value,

        -- leg role within a spread: SHORT leg receives premium, LONG leg pays
        case
            when direction = 'SOLD'   and is_opening then 'SHORT_OPEN'
            when direction = 'BOUGHT' and is_opening then 'LONG_OPEN'
            when direction = 'BOUGHT' and is_closing then 'SHORT_CLOSE'
            when direction = 'SOLD'   and is_closing then 'LONG_CLOSE'
            when transaction_type = 'EXPIRED'        then 'EXPIRED'
            when transaction_type = 'ASSIGNED'       then 'ASSIGNED'
            else 'UNKNOWN'
        end                                                         as leg_role,

        -- calendar fields for aggregation
        date_trunc('month', trade_date)                             as trade_month,
        date_trunc('quarter', trade_date)                           as trade_quarter,
        extract('year' from trade_date)                             as trade_year

    from stg
)

select * from enriched

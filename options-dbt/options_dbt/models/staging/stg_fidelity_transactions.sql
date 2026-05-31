{{
    config(
        materialized='view',
        description='Cleaned and parsed Fidelity option transaction legs. One row = one option contract leg.'
    )
}}

with source as (
    select
        *,
        row_number() over (
            partition by run_date, account, symbol, action, amount
            order by (select null)
        ) as _row_num
    from {{ ref('fidelity_transactions') }}
),

parsed as (
    select
        -- keys: include row_num to handle same-day duplicate legs (e.g., 2 contracts on same spread)
        md5(concat_ws('|',
            cast(run_date as varchar),
            account,
            coalesce(symbol, ''),
            action,
            cast(coalesce(amount, 0) as varchar),
            cast(_row_num as varchar)
        ))                                                              as transaction_id,

        -- dates
        run_date                                                            as trade_date,
        settlement_date,

        -- account
        account                                                             as account_name,

        -- transaction classification parsed from the action string
        case
            when action ilike '%OPENING%'  then 'OPENING'
            when action ilike '%CLOSING%'  then 'CLOSING'
            when action ilike '%EXPIRED%'  then 'EXPIRED'
            when action ilike '%ASSIGNED%' then 'ASSIGNED'
            when action ilike '%DIVIDEND%' then 'DIVIDEND'
            else 'OTHER'
        end                                                                 as transaction_type,

        case
            when action ilike '%YOU SOLD%'   then 'SOLD'
            when action ilike '%YOU BOUGHT%' then 'BOUGHT'
            else null
        end                                                                 as direction,

        -- option metadata parsed from OCC symbol: -COST260402P940
        regexp_extract(symbol, '-?([A-Z\.]+)\d{6}[PC][\d\.]+', 1)         as ticker,

        case
            when regexp_extract(symbol, '-?[A-Z\.]+(\d{2})(\d{2})(\d{2})[PC]', 1) != ''
                then strptime(
                    '20' ||
                    regexp_extract(symbol, '-?[A-Z\.]+(\d{6})[PC]', 1),
                    '%Y%m%d'
                )::date
            else null
        end                                                                 as expiry_date,

        case
            when symbol ilike '%P%' and action ilike '%PUT%'  then 'PUT'
            when symbol ilike '%C%' and action ilike '%CALL%' then 'CALL'
            else null
        end                                                                 as option_type,

        cast(
            regexp_extract(symbol, '-?[A-Z\.]+\d{6}[PC]([\d\.]+)', 1)
            as double
        )                                                                   as strike,

        symbol                                                              as symbol_raw,
        action                                                              as action_raw,
        type                                                                as margin_type,

        -- financials
        abs(quantity)                                                       as contracts,
        price                                                               as price_per_share,
        coalesce(price, 0) * 100                                           as premium_per_contract,
        coalesce(commission, 0)                                             as commission,
        coalesce(fees, 0)                                                   as fees,
        coalesce(commission, 0) + coalesce(fees, 0)                        as total_cost,
        coalesce(amount, 0)                                                 as amount,

        -- derived flags
        case when action ilike '%OPENING%' then true else false end         as is_opening,
        case when action ilike '%CLOSING%'
              or action ilike '%EXPIRED%'
              or action ilike '%ASSIGNED%'
            then true else false end                                        as is_closing,

        -- category for filtering
        case
            when action ilike '%DIVIDEND%' then 'DIVIDEND'
            when action ilike '%ASSIGNED%' then 'ASSIGNMENT'
            else 'OPTION'
        end                                                                 as transaction_category

    from source
    where run_date is not null
)

select * from parsed

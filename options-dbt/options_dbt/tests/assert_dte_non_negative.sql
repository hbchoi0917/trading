-- Days-to-expiry at time of trade must be >= 0.
-- A negative DTE means the trade was recorded after expiry — data error.
-- Returns violating rows (test fails if any rows returned).

select
    transaction_id,
    ticker,
    trade_date,
    expiry_date,
    dte_at_trade
from {{ ref('int_option_legs') }}
where
    dte_at_trade is not null
    and dte_at_trade < 0

-- Days-to-expiry at time of trade must be >= 0.
-- A negative DTE on an opening/closing trade means it was recorded after expiry — data error.
-- EXPIRED / ASSIGNED rows are excluded: they are broker postings, not trades, and Fidelity
-- books them on the next business day (e.g. Friday expiry posted Monday → DTE -3).
-- Returns violating rows (test fails if any rows returned).

select
    transaction_id,
    ticker,
    transaction_type,
    trade_date,
    expiry_date,
    dte_at_trade
from {{ ref('int_option_legs') }}
where
    dte_at_trade is not null
    and dte_at_trade < 0
    and transaction_type not in ('EXPIRED', 'ASSIGNED')

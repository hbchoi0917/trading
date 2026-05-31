-- For closed spreads, close_date must be on or after open_date.
-- A negative holding period indicates a data integrity issue.
-- Returns violating rows (test fails if any rows returned).

select
    spread_id,
    ticker,
    open_date,
    close_date,
    holding_days
from {{ ref('int_spread_trades') }}
where
    not is_open
    and close_date < open_date

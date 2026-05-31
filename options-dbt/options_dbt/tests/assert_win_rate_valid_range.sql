-- Win rate percentages must be between 0 and 100 (inclusive).
-- Checks both mart_ticker_performance and mart_monthly_pnl.
-- Returns violating rows (test fails if any rows returned).

select 'ticker' as source, ticker as entity, win_rate_pct
from {{ ref('mart_ticker_performance') }}
where win_rate_pct < 0 or win_rate_pct > 100

union all

select 'monthly' as source, account_name as entity, win_rate_pct
from {{ ref('mart_monthly_pnl') }}
where win_rate_pct < 0 or win_rate_pct > 100

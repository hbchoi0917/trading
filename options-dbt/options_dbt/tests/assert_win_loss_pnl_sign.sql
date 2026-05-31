-- Trade result must be consistent with realized P&L sign.
-- WIN  → realized_pnl > 0
-- LOSS → realized_pnl < 0
-- Returns rows that violate this invariant (test fails if any rows returned).

select
    spread_id,
    ticker,
    trade_result,
    realized_pnl
from {{ ref('int_spread_trades') }}
where
    (trade_result = 'WIN'  and realized_pnl <= 0)
    or (trade_result = 'LOSS' and realized_pnl >= 0)

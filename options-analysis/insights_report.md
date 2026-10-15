# Options Trading Insights Report
**Period:** January 2025 – September 2026 (21 months, manually traded) · **Basis:** realized P&L
**Primary Strategy:** Put Credit Spreads + Covered Calls + Wheel

> Every result in this report is **realized P&L**: each close is matched to its
> opening trade (FIFO, per account and contract) and booked when the position
> closed. Option legs only — the stock side of assignments and covered calls,
> dividends and interest are not included. Results are given as direction,
> rank and ratio; specific dollar figures are kept in a private repository.
> An earlier version of this report (May 2026) was written on a net-cash-flow
> basis; see *Why realized P&L* in the [README](README.md).

---

## 1. Overall Performance

| | 2025 | 2026 (Jan–Sep) |
|---|---|---|
| **Realized P&L** | Profitable | Profitable — already **~4x** full-year 2025 |
| **Shape of the year** | Loss in H1, strong H2 | Loss in Q1, then the two strongest quarters on record |
| **Primary strategy** | Put credit spreads | Put credit spreads |

- **Puts carry the book.** Put-side realized P&L is strongly positive; the call
  side (mostly covered calls, option leg only) is a net loss.
- **Concentration.** COST alone is roughly a third of net realized P&L, and the
  top three names (COST, NVDA, MU) a little over half. The anchor works, and it
  is also the largest single-name dependency in the book.

---

## 2. Quarterly Trend

| Quarter | Direction | Notes |
|---|---|---|
| 2025 Q1 | ↘ | Small net loss; February drag |
| 2025 Q2 | ↘ | April tariff-driven drawdown — worst month of 2025 |
| 2025 Q3 | ↗ | Recovery |
| 2025 Q4 | ↗↗ | Strongest quarter of 2025 |
| 2026 Q1 | ↘ | February MSFT / NFLX blow-ups — worst month on record |
| 2026 Q2 | ↗↗↗ | Second-strongest quarter on record; May led |
| **2026 Q3** | ↗↗↗ | **Strongest quarter on record; August the best month on record** |

Both years follow the same pattern: an early-year loss driven by one event
(the April 2025 tariff drawdown, the February 2026 single-name blow-ups),
followed by a strong recovery. The recovery in 2026 was larger and came faster.

---

## 3. Ticker Results (21 months, realized)

### Largest contributors

In order: **COST, NVDA, MU, SOXX, STX, CLS, HIMS, AVGO, SPXW, DRAM, IWM, MARA.**

| Ticker | Notes |
|---|---|
| COST | Core anchor; steady, wide-strike put spreads across accounts |
| NVDA | Most active name; consistent put spread winner |
| MU | Semiconductor cycle; timing aligned with the recovery |
| SOXX | Positive overall, though covered-call closes gave back part of the premium |
| CLS | Recovered fully from its 2025 losses on narrow spreads |
| HIMS | High volatility, high premium relative to risk |
| SPXW / IWM | Index put spreads — no early-assignment risk |
| DRAM | Rotation confirmed with both put spreads and covered calls |
| MARA | Wheel (cash-secured put → assignment → covered call) net positive |

### Largest detractors (worst first)

| Ticker | Root cause |
|---|---|
| MSFT | Sharp, idiosyncratic February 2026 drop with no broad-market catalyst. The stop was applied later than optimal and repeated rolls compounded the loss. Lesson: exit faster on single-name drops with no sector confirmation. **Excluded from rotation.** |
| NFLX | Event-driven decline on post-split M&A speculation — outside normal options risk management. (The 10-for-1 split is handled in the data by matching pre- and post-split strikes as one contract.) **Excluded from rotation.** |
| EWY | Covered calls during the KRW/EWY rally: buy-to-close cost exceeded the premium. Option leg only — gains on the shares are not in this figure. |
| AMAT | Includes a position still open at the end of September; not final. |
| RGTI | Quantum-computing volatility; spreads too narrow. **Excluded from rotation.** |
| PLTR | Calls sold too close to the money during the 2025 rally, then put-spread rolls. Improved in early 2026 but still a net loss over the period. |
| IONQ | High-volatility small cap; sharp February–March drop forced an early close. **Excluded from rotation.** |
| INTC | Short-dated straddle-style trades after a catalyst. The May review read these as working; on a realized basis through September the name is a small net loss — revisit before repeating. |

---

## 4. Key Insights

### ✅ What's working
1. **COST as core anchor** — the largest contributor by a wide margin. Watch
   the concentration that comes with it.
2. **Index put spreads (SPXW, IWM)** — positive with no early-assignment risk.
3. **Exclusion discipline** — MSFT, NFLX, IONQ and RGTI were removed after
   confirmed blow-ups. All four rank among the largest detractors, which
   validates the exclusions.
4. **Recovery speed** — the 2026 drawdown was deeper than 2025's, and the
   recovery that followed was larger and faster.
5. **Wheel on high-premium names** — MARA validated the approach with
   acceptable assignment risk.

### ⚠️ Lessons: covered calls
Calls are the weak side of the book (net loss, option leg only).
1. **GOOGL** — buy-to-close cost exceeded the initial premium in a rally;
   rolling forward at the same strike recovers the drag through cumulative
   premium.
2. **EWY** — holding through the rally and selling the stock above the strike
   beat an early buy-to-close.
3. **QQQM** — absorb the buy-to-close, sell the stock above the strike, rebuy
   the shares, re-sell the call; track option and stock P&L separately.
4. **Account concentration** — covered-call drag sat mostly in one account;
   shifting that allocation toward put spreads balances it.

### ⚠️ Lessons: spread management
1. **Single-name drops** — exit faster when a name falls with no sector
   confirmation (MSFT).
2. **TSLA sizing** — repeated put spread cycles caught downside moves; reduce
   contract count or widen strikes, and avoid covered calls.
3. **PLTR strike buffer** — volatility needs a wider OTM buffer.
4. **Short-dated straddles (INTC)** — keep opportunistic and small until the
   realized result turns positive.

---

## 5. Standing Rules

1. **Exclusions** — MSFT, NFLX, IONQ, RGTI stay out of rotation.
2. **Max risk per spread** — spread width × contracts × 100, capped per position.
3. **DTE discipline** — open at 28–45 DTE; close at the profit target or the DTE
   threshold, whichever comes first.
4. **Drawdown circuit breaker** — pause new positions when the monthly loss
   threshold is hit.
5. **High-beta names** — wheel only, with a small contract cap.
6. **Call selling** — only sell calls when the stock is below its 52-week midpoint.

---

## Data Notes

- Built from Fidelity transaction exports. Months where a quarterly export
  silently omitted rows were rebuilt from monthly re-exports, and duplicate
  rows were resolved without dropping legitimate split fills.
- Closes for positions opened before January 2025 are excluded from realized
  P&L.
- A few contracts with no expiration record were treated as expired worthless.
- Positions still open at the end of September are not in realized P&L; they
  were reconciled against the broker's position screen.
- Pre-tax figures; not a tax calculation.

*Report updated October 2026 from exports through September 30, 2026.*

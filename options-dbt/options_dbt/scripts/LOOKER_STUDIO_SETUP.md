# Looker Studio Dashboard Setup Guide

## 1. Google Sheets 준비

1. Google Sheets에서 새 스프레드시트 생성
2. 시트 4개 수동 생성 (또는 `export_to_sheets.py` 실행):
   - `mart_monthly_pnl`
   - `mart_ticker_performance`
   - `mart_account_summary`
   - `int_spread_trades`
3. `looker_exports/` 폴더의 CSV 파일을 각 시트에 붙여넣기

## 2. Looker Studio 연결

1. [Looker Studio](https://lookerstudio.google.com) → 새 보고서
2. 데이터 소스 추가 → Google Sheets → 위 스프레드시트 선택
3. 시트 4개를 각각 데이터 소스로 추가

## 3. 대시보드 페이지 구성 (3페이지 권장)

---

### Page 1: Portfolio Overview

**목적:** 첫눈에 전체 성과를 파악 (BI 인터뷰용 임팩트 극대화)

| 차트 | 타입 | 데이터소스 | 설정 |
|---|---|---|---|
| Lifetime P&L | Scorecard | mart_account_summary | SUM(lifetime_pnl), 색상: 녹색 |
| Monthly Win Rate | Scorecard | mart_monthly_pnl | AVG(win_rate_pct) |
| Total Trades | Scorecard | mart_account_summary | SUM(total_trades) |
| Best Month | Scorecard | mart_monthly_pnl | MAX(net_pnl) |
| 월별 P&L + 누적 (콤보) | Combo Chart | mart_monthly_pnl | Bar=net_pnl, Line=cumulative_pnl, Dimension=month |
| 계좌별 누적 P&L | Stacked Bar | mart_monthly_pnl | Dimension=month, Breakdown=account_name, Metric=net_pnl |

**필터:** Account 드롭다운 (account_name 기준)

---

### Page 2: Ticker Performance

**목적:** "어떤 종목에서 돈 벌었나" — DA/AE 인터뷰의 핵심 스토리

| 차트 | 타입 | 데이터소스 | 설정 |
|---|---|---|---|
| 티커별 총 P&L | Bar Chart (수평) | mart_ticker_performance | Dimension=ticker, Metric=total_pnl, Sort DESC, Top 15 |
| 승률 vs P&L 산점도 | Scatter Chart | mart_ticker_performance | X=win_rate_pct, Y=total_pnl, Size=total_trades, Color=is_excluded |
| 거래 빈도 vs 효율성 | Bubble Chart | mart_ticker_performance | X=total_trades, Y=avg_pnl_per_trade, Size=total_pnl |
| 제외 티커 하이라이트 | Table | mart_ticker_performance | Filter: is_excluded=true, 컬럼: ticker/total_pnl/total_trades |

**인사이트 텍스트 박스 추가:**
> "MSFT (-$10,556) and NFLX (-$6,238) were excluded from rotation after Q1 2026 blow-ups.
> Remaining 48 tickers show 50%+ win rate on average."

---

### Page 3: Trade Detail & Risk

**목적:** "데이터를 얼마나 깊이 이해하는가" — 모든 포지션 레벨 인터뷰용

| 차트 | 타입 | 데이터소스 | 설정 |
|---|---|---|---|
| 월별 거래수 | Bar Chart | mart_monthly_pnl | Dimension=month, Metric=trade_count |
| 평균 보유일 트렌드 | Line Chart | int_spread_trades | Dimension=open_month, Metric=AVG(holding_days) |
| Close Reason 분포 | Pie Chart | int_spread_trades | Dimension=close_reason, Metric=COUNT |
| 개별 트레이드 테이블 | Table | int_spread_trades | 컬럼: ticker/account/open_date/close_date/realized_pnl/close_reason |

**필터:** ticker 검색바, date range picker, account 드롭다운

---

## 4. 스타일 권장사항

- **색상 팔레트:** 수익 = #00C851 (green), 손실 = #FF4444 (red), 중립 = #4A90D9 (blue)
- **폰트:** 제목 Google Sans Bold, 본문 Roboto
- **배경:** #F8F9FA (연한 회색) — 흰색보다 professional해 보임
- **로고/제목:** "Options Trading Analytics | Jan 2025 – May 2026" 헤더 추가

---

## 5. 공유 설정

포트폴리오용 공개 링크 생성:
- 보고서 우측 상단 → Share → Manage access → **Anyone with the link can view**
- 이 링크를 이력서/LinkedIn/GitHub README에 추가

---

## 6. 인터뷰별 강조 포인트

| 포지션 | 보여줄 페이지 | 핵심 멘트 |
|---|---|---|
| **AE** | Page 1 + "dbt lineage가 이 숫자를 만들었다" | 파이프라인 설계, surrogate key, spread matching 로직 |
| **Senior DA** | Page 2 | "MSFT 손실을 어떻게 감지했고 rotation에서 제외했나" |
| **BI Developer** | Page 1 + Page 3 | "mart → Sheets → Looker Studio 전체 스택을 혼자 구축했다" |
| **BA** | Page 1 스크린샷만 | "$48,995 lifetime P&L, 78% monthly win rate" — 숫자로 말하기 |

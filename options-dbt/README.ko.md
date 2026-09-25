# 옵션 트레이딩 dbt 파이프라인

**Fidelity 거래내역 원본 export를 스프레드 단위 손익으로 변환하는 dbt + DuckDB 파이프라인입니다.**

[![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?logo=streamlit&logoColor=white)](https://options-trading-dash.streamlit.app/)

[English](README.md) | **한국어**

---

## 해결하려는 문제

Fidelity는 **거래 레그** 단위(계약 액션 하나당 한 행)로만 데이터를 내보냅니다. "스프레드 거래"나 거래 사이클별 실현손익이라는 개념 자체가 없습니다. 풋 크레딧 스프레드 하나만 해도 최소 두 개의 레그(진입 + 청산)가 생기고, 롤까지 하면 네 개 이상이 됩니다.

이 파이프라인은 다음 순서로 이 문제를 풉니다.

1. OCC 옵션 심볼 파싱 (`-COST260402P940` → 종목 `COST`, 만기 `2026-04-02`, 행사가 `940`, 유형 `PUT`)
2. 자연키(계좌 + 종목 + 만기 + 옵션 유형 + 행사가)로 레그를 묶어 스프레드 거래 사이클로 구성
3. 사이클별 실현손익, 승/패 분류, 보유 일수, 진입 시점 DTE 계산
4. 월별·종목별·계좌별 마트 테이블로 집계
5. 마트에서 바로 라이브 Streamlit 대시보드로 연결

---

## 파이프라인 개요

```
Fidelity CSV export
        │
        ▼
┌─────────────────────────────┐
│  STAGING                    │  파싱, 정제, 타입 변환
│  stg_fidelity_transactions  │  한 행 = 계약 레그 하나
└────────────────┬────────────┘
                 │
        ┌────────▼────────┐
        │  INTERMEDIATE   │  비즈니스 로직 적용
        │  int_option_legs│  DTE, 명목금액, 레그 역할 추가
        │  int_spread_    │  레그 → 스프레드 거래 사이클로 그룹핑
        │  trades         │  사이클별 실현손익 계산
        └────────┬────────┘
                 │
   ┌─────────────┼──────────────┐
   ▼             ▼              ▼
mart_monthly  mart_ticker   mart_account
_pnl          _performance  _summary
   │             │              │
   └─────────────┴──────────────┘
                 │
                 ▼
        Streamlit 대시보드
```

---

## 레이어별 설명

### Staging — "원천 데이터 정제만 한다"

**입력:** Fidelity CSV 원본
```
run_date   | action                                     | symbol         | amount
2025-03-31 | YOU SOLD OPENING TRANSACTION PUT (PLTR)... | -PLTR250502P82 | 670.30
```

**Staging에서 하는 일:** 컬럼명 변경, 타입 변환, 심볼 파싱만 합니다. 비즈니스 로직은 넣지 않습니다.

```sql
-- OCC 심볼에서 종목, 만기, 행사가, 옵션 유형 추출
regexp_extract(symbol, '-?([A-Z]+)\d{6}[PC]', 1)        as ticker       -- 'PLTR'
strptime('20' || regexp_extract(symbol, '\d{6}', 0), '%Y%m%d') as expiry_date  -- 2025-05-02
cast(regexp_extract(symbol, '[PC]([\d\.]+)', 1) as double)     as strike        -- 82.0
case when action ilike '%OPENING%' then 'OPENING' end          as transaction_type
case when action ilike '%YOU SOLD%' then 'SOLD'   end          as direction
```

**출력:** 한 행 = 계약 레그 하나. 해석 없이 원천에 충실합니다.

---

### Intermediate — "비즈니스 로직 적용"

**핵심 문제:** Fidelity는 레그를 주는데, 필요한 건 스프레드입니다.

```
leg 1: PLTR PUT 82 | SOLD OPENING   | +670.30  ← 진입, 프리미엄 수령
leg 2: PLTR PUT 82 | BOUGHT CLOSING | -120.00  ← 청산, 되사기
              ↓ (계좌 + 종목 + 만기 + 옵션 유형 + 행사가)로 그룹핑
spread: PLTR PUT 82 | realized_pnl = +550.30 | WIN | 32일 보유
```

**`int_spread_trades.sql`에 적어둔 설계 결정:**
> 레그를 순서대로 짝짓지 않고 자연키로 묶습니다. 그래서 롤도 자연스럽게 처리됩니다. 같은 키에서 CLOSING 레그 뒤에 새 OPENING 레그가 오면 두 개의 별도 사이클 행이 됩니다.

**`int_option_legs.sql`에서 추가하는 컬럼:**

| 컬럼 | 계산 | 용도 |
|---|---|---|
| `dte_at_trade` | `expiry_date - trade_date` | 진입 DTE 규칙 준수 분석 |
| `notional_value` | `strike × 100 × contracts` | 리스크 노출 규모 파악 |
| `leg_role` | `SOLD + OPENING → SHORT_OPEN` | 스프레드 구조 점검 |

**출력:** 한 행 = 완결된 스프레드 거래 사이클 하나 (실현손익, 승/패, 보유 일수 포함)

---

### Marts — "비즈니스 질문에 바로 답한다"

Intermediate는 거래 단위 행을 주고, 마트는 비즈니스 질문에 바로 답합니다.

**`mart_monthly_pnl`** — *"계좌별로 매달 성과가 어땠나?"*
```sql
sum(realized_pnl)                                            as net_pnl
sum(net_pnl) over (partition by account_name order by month) as cumulative_pnl
net_pnl - lag(net_pnl) over (partition by account_name order by month) as pnl_mom_change
case when net_pnl > 0 then true else false end               as is_win_month
```

**`mart_ticker_performance`** — *"어떤 종목을 로테이션에 넣을 가치가 있나?"*
```sql
sum(realized_pnl)                          as total_pnl
count(*) filter (trade_result = 'WIN')     as wins
row_number() over (order by total_pnl desc, ticker, option_type) as pnl_rank
case when ticker in ('MSFT','NFLX','IONQ','RGTI') then true end as is_excluded
```

**`mart_account_summary`** — *"계좌별 누적 성과는 어떻게 비교되나?"*
- 거래 단위 승률과 월 단위 승률 (서로 다른 지표)
- 계좌별 최고/최저 단일 거래와 최고/최저 월
- `int_spread_trades`와 `mart_monthly_pnl`을 조인해 두 기준을 모두 계산

**출력:** Streamlit 대시보드가 `SELECT * FROM main_marts.mart_*`로 마트를 직접 조회합니다.

---

## 가능해진 분석

Fidelity 원본 export로는 **답할 수 없던** 질문을 이제 바로 쿼리할 수 있습니다.

| 질문 | 이전 | 이후 |
|---|---|---|
| 이 스프레드의 실제 수익은? | ❌ 레그만 있음 | ✅ 사이클별 `realized_pnl` |
| 계좌별 월간 승률은? | ❌ | ✅ `mart_monthly_pnl.is_win_month` |
| 어떤 DTE 구간이 가장 수익성이 좋나? | ❌ | ✅ 종목 마트의 `avg_dte_at_open` |
| COST vs TSLA, 거래당 효율은 어느 쪽이 좋나? | ❌ | ✅ `pnl_per_trade` + `pnl_rank` |
| MSFT를 다시 거래해야 하나? | ❌ | ✅ 손실 이력이 담긴 `is_excluded` 플래그 |

---

## 포트폴리오 관점에서의 의미

> *"Fidelity 원본 export는 거래 레그만 줍니다. 이 파이프라인은 레그를 거래로, 거래를 월별 손익으로, 월별 손익을 포트폴리오 의사결정으로 바꾸고, 라이브 대시보드로 결과를 전달합니다."*

이 프로젝트는 애널리틱스 엔지니어링의 전 과정을 보여줍니다.

1. **지저분한 원천 데이터** → OCC 심볼이 파싱되지 않고 거래 ID도 없는 실제 증권사 export
2. **간단하지 않은 변환** → 명시적 외래키 없이 레그를 스프레드로 매칭
3. **레이어드 모델링** → staging / intermediate / marts로 역할을 명확히 분리
4. **데이터 품질** → 스키마 테스트(unique, not\_null, accepted\_values) + SQL 단일 테스트 4개
5. **결과 전달** → 라이브 Streamlit 대시보드, `export_to_sheets.py`를 통한 Google Sheets export

---

## 프로젝트 구조

```
options-dbt/
├── options_dbt/                    ← dbt 프로젝트 루트
│   ├── models/
│   │   ├── staging/
│   │   │   ├── stg_fidelity_transactions.sql   OCC 심볼 파싱, 레그 분류
│   │   │   ├── _sources.yml
│   │   │   └── _stg_models.yml                 스키마 테스트
│   │   ├── intermediate/
│   │   │   ├── int_option_legs.sql             레그 보강 (DTE, 명목금액, 역할)
│   │   │   ├── int_spread_trades.sql           레그 → 스프레드 손익 사이클
│   │   │   └── _int_models.yml                 스키마 테스트
│   │   └── marts/
│   │       ├── mart_monthly_pnl.sql
│   │       ├── mart_ticker_performance.sql
│   │       ├── mart_account_summary.sql
│   │       └── _mart_models.yml                스키마 테스트
│   ├── tests/
│   │   ├── assert_win_loss_pnl_sign.sql        WIN → pnl > 0, LOSS → pnl < 0
│   │   ├── assert_close_after_open.sql         close_date ≥ open_date
│   │   ├── assert_dte_non_negative.sql         진입·청산 레그의 DTE ≥ 0
│   │   └── assert_win_rate_valid_range.sql     win_rate_pct가 0~100 범위
│   ├── seeds/
│   │   └── sample_fidelity_transactions.csv   dbt seed용 합성 샘플 데이터
│   ├── scripts/
│   │   ├── export_to_sheets.py                DuckDB → Google Sheets → Looker Studio
│   │   └── LOOKER_STUDIO_SETUP.md
│   ├── dbt_project.yml
│   └── profiles.yml                           env_var로 DuckDB 경로 지정
└── streamlit_app/
    ├── app.py                                 3페이지 인터랙티브 대시보드
    ├── sample_options_trading.duckdb          미리 빌드한 마트 (샘플 거래 162건)
    └── requirements.txt
```

---

## 빠른 시작

```bash
# 1. dbt-duckdb 설치
pip install dbt-core dbt-duckdb

# 2. 샘플 데이터 로드 후 모델 빌드
cd options-dbt/options_dbt
dbt seed          # sample_fidelity_transactions.csv 로드
dbt run           # 모든 모델 빌드
dbt test          # 스키마 + 단일 테스트 실행

# 3. 대시보드 실행
cd ../streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

---

## 스택

- **[dbt-core](https://docs.getdbt.com/)** — 모델 오케스트레이션, 스키마 테스트, 문서화
- **[DuckDB](https://duckdb.org/)** — 서버 없이 돌아가는 임베디드 OLAP 엔진
- **[Streamlit](https://streamlit.io/)** — 대시보드 배포
- **[Plotly](https://plotly.com/)** — 인터랙티브 차트
- **[pandas](https://pandas.pydata.org/)** — DuckDB와 Streamlit 사이의 DataFrame 레이어

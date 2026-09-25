# 옵션 트레이딩 dbt 프로젝트

[English](README.md) | **한국어**

실제 옵션 트레이딩 포트폴리오를 위한 분석 파이프라인입니다. 18개월간의 실거래 데이터(2025년 1월 – 2026년 5월), 4개 증권 계좌, 50개 종목을 다룹니다.

원천 데이터: Fidelity 거래내역 export (옵션 레그만)

---

## 스택

| 레이어 | 도구 |
|---|---|
| 변환 | dbt-core 1.11 |
| 웨어하우스 | DuckDB (로컬, 별도 인프라 없음) |
| 원천 데이터 | Fidelity CSV export (옵션 거래) |

---

## 프로젝트 구조

```
models/
├── staging/
│   └── stg_fidelity_transactions   # OCC 심볼 파싱 → 종목/만기/행사가/옵션 유형
├── intermediate/
│   ├── int_option_legs              # 레그 보강: DTE, 레그 역할, 명목금액
│   └── int_spread_trades            # 레그 → 스프레드 사이클별 실현손익
└── marts/
    ├── mart_monthly_pnl             # 계좌별 월간 손익 + 누적
    ├── mart_ticker_performance      # 종목별 승률, 평균 손익, 거래 빈도
    └── mart_account_summary         # 계좌별 누적 성과
```

---

## 주요 설계 결정

**row_number 기반 대리키:** 같은 날 같은 가격에 여러 계약을 거래하면 Fidelity export에 완전히 똑같은 행이 생길 수 있습니다. `row_number()` 파티션으로 정상 데이터를 버리지 않고 중복을 구분합니다.

**자연키 기반 스프레드 매칭:** 레그를 순서대로 짝짓지 않고 `(account, ticker, expiry, option_type, strike)`로 묶습니다. 그래서 롤된 포지션도 제대로 처리됩니다. CLOSING 레그 뒤에 새 OPENING 레그가 오면 두 개의 별도 거래 사이클이 됩니다.

**배당·배정 분리:** `stg_fidelity_transactions`의 `transaction_category` 플래그로 옵션이 아닌 현금흐름을 스프레드 매칭 전에 걸러내, 손익이 오염되지 않게 합니다.

---

## 설정

```bash
pip install dbt-duckdb

# 샘플 데이터로 실행 (익명화된 37행, 엣지 케이스 전부 포함)
dbt build --profiles-dir .

# 전체 export로 실행 (seeds/fidelity_transactions.csv, gitignore 대상)
dbt build --profiles-dir . --vars '{transactions_seed: fidelity_transactions}'
```

입력 seed는 `transactions_seed` 변수로 고릅니다(기본값: `sample_fidelity_transactions`). 샘플로 빌드할 때는 매번 로그에 안내 문구가 찍힙니다.

---

## 데이터 품질 테스트 (총 48개, 전부 통과)

모든 레이어의 스키마 테스트 44개(`unique`, `not_null`, `accepted_values`)와 `tests/`의 SQL 단일 테스트 4개로 구성됩니다. staging 레이어 예시는 다음과 같습니다.

- `transaction_id`에 `unique` + `not_null`
- `trade_date`, `account_name`, `transaction_type`, `amount`에 `not_null`
- `account_name`, `transaction_type`, `transaction_category`, `option_type`에 `accepted_values`

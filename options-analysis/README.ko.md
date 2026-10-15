# 옵션 트레이딩 분석 (2025년 1월 – 2026년 9월)

[English](README.md) | **한국어**

## 개요

4개 증권 계좌(Account A, B, C, D)에서 직접(수동으로) 매매한 개인 옵션 거래 내역(2025년 1월 – 2026년 9월, 21개월)을 분석한 파이프라인과, 거기서 얻은 인사이트를 담은 폴더입니다.

---

## 폴더 구조

```
options-analysis/
├── data/
│   ├── Accounts_History*.csv    # Fidelity 원본 export (직접 추가, 커밋 금지)
│   ├── options_cleaned.csv      # analysis.py가 자동 생성
│   └── realized_pnl.csv         # 자동 생성: FIFO로 매칭된 청산 1건당 1행
├── charts/                      # PNG 차트 9종 — 샘플만 커밋
├── analysis.py                  # 전체 파이프라인: 로드 → 정제 → 분석 → 시각화
├── generate_sample_charts.py    # 합성 데이터로 charts/ 생성 (공개용)
├── insights_report.md           # 핵심 발견 & 액션 플랜
└── README.md
```

---

## 주요 지표 (2025년 1월 – 2026년 9월)

모든 수치는 **실현손익** 기준입니다. 청산 거래를 각각의 진입 거래와 매칭(계좌·계약별 FIFO)해서, 포지션을 닫은 시점에 손익을 기록합니다.

| 지표 | 값 |
|---|---|
| 연도별 실현손익 | **두 해 모두 수익** (2025년, 2026년 9월까지) |
| 전년 대비 | 2026년은 9월까지만으로 이미 2025년 전체의 **약 4배** |
| 최근 분기 | **2026년 2분기와 3분기**가 역대 가장 좋은 두 분기 |
| 주력 전략 | 풋 크레딧 스프레드 |
| 계좌 | 4개 (일반 계좌 + IRA) |
| 기간 | 2025년 1월 – 2026년 9월 |

**현금흐름이 아니라 실현손익인 이유.** 거래일 기준으로 들어오고 나간 프리미엄을 더하면 순현금흐름이지 손익이 아닙니다. 크레딧 스프레드는 프리미엄을 먼저 받기 때문에, 신규 진입이 많은 달은 결과가 나오기도 전에 수익처럼 보이고, 같은 달에 현금흐름과 실현손익의 방향이 반대로 나오기도 합니다. (이 README의 이전 버전에는 현금흐름 기준 승률이 적혀 있었습니다.)

> 구체적인 손익 수치는 private 레포에서 관리합니다.

---

## 실행 방법

```bash
# 1. 의존성 설치
pip install pandas plotly kaleido

# 2. Fidelity CSV export를 data/에 복사
cp ~/Downloads/Accounts_History*.csv data/

# 3. 전체 파이프라인 실행
python analysis.py
# → 실현손익(FIFO)과 순현금흐름을 나란히 출력하고,
#   실현손익 + 미청산 프리미엄 + 기간 이전 청산 == 순현금흐름 인지 검증
# → data/options_cleaned.csv, data/realized_pnl.csv 저장
# → charts/에 차트 9종 저장 (손익 차트는 실현손익 기준)
```

**기간이 겹치는 export를 함께 넣어도 됩니다.** 여러 export에 반복된 행은 한 번만 집계하지만, 한 export *안의* 똑같은 행은 그대로 둡니다. 분할 체결(같은 계약·가격·날짜)은 똑같은 행으로 찍히는데 실제 거래이기 때문에, 단순히 중복을 일괄 제거하면 이런 거래가 소리 없이 사라집니다.

---

## 샘플 차트

> 아래 차트는 [`insights_report.md`](insights_report.md)(영문)의 흐름을 반영한 합성 데이터로 만든 것이며,
> 실제 손익 수치는 private 레포에 있습니다. 본인 데이터로 만들려면 `data/`에 CSV export를 넣고
> `python analysis.py`를 실행하세요.

### 월별 손익 + 누적 성장
![월별 손익](charts/chart1_monthly_pnl.png)

### 종목별 순손익
![종목별 손익](charts/chart2_ticker_pnl.png)

### 분기별 추이
![분기별 손익](charts/chart8_quarterly_pnl.png)

### 종목 효율 (거래당 손익 vs 거래 빈도)
![종목 효율](charts/chart9_ticker_efficiency.png)

### 전략 비중 — PUT vs CALL
![PUT vs CALL](charts/chart3_put_vs_call.png)

<details>
<summary>전체 차트 9종</summary>

| 파일 | 설명 |
|---|---|
| `chart1_monthly_pnl.png` | 월별 손익 막대 + 누적 성장 곡선 |
| `chart2_ticker_pnl.png` | 종목별 순손익 (상위 수익·손실 종목) |
| `chart3_put_vs_call.png` | PUT vs CALL 전략별 손익 |
| `chart4_trade_count.png` | 월별 거래 빈도 추이 |
| `chart5_account_pnl.png` | 계좌 유형별 순손익 |
| `chart6_weekday_pnl.png` | 요일별 손익 |
| `chart7_ticker_frequency.png` | 종목별 신규 진입 건수 |
| `chart8_quarterly_pnl.png` | 분기별 손익 막대 + 누적 성장 곡선 |
| `chart9_ticker_efficiency.png` | 거래당 손익 vs 거래 빈도 산점도 |

</details>

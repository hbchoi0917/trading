# 옵션 트레이딩 분석 (2025년 1월 – 2026년 5월)

[English](README.md) | **한국어**

## 개요

4개 증권 계좌(Account A, B, C, D)의 개인 옵션 거래 내역(2025년 1월 1일 – 2026년 5월 29일)을 분석한 파이프라인과, 거기서 얻은 인사이트를 담은 폴더입니다.

---

## 폴더 구조

```
options-analysis/
├── data/
│   ├── Accounts_History*.csv    # Fidelity 원본 export (직접 추가, 커밋 금지)
│   └── options_cleaned.csv      # analysis.py가 자동 생성
├── charts/                      # PNG 차트 9종 — 샘플만 커밋
├── analysis.py                  # 전체 파이프라인: 로드 → 정제 → 분석 → 시각화
├── generate_sample_charts.py    # 합성 데이터로 charts/ 생성 (공개용)
├── insights_report.md           # 핵심 발견 & 액션 플랜
└── README.md
```

---

## 주요 지표 (2025년 1월 – 2026년 5월)

| 지표 | 값 |
|---|---|
| 월간 승률 | **78%** (18개월 중 14개월 수익) |
| 주력 전략 | 풋 크레딧 스프레드 |
| 계좌 | 4개 (일반 계좌 + IRA) |
| 기간 | 2025년 1월 – 2026년 5월 |

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
# → 요약 통계를 콘솔에 출력
# → data/options_cleaned.csv 저장
# → charts/에 차트 9종 저장
```

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

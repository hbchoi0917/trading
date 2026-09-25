# 옵션 트레이딩 분석 대시보드

[English](README.md) | **한국어**

dbt 분석 파이프라인 위에 Streamlit + DuckDB + Plotly로 만든 대시보드입니다.

**데이터:** 공개 데모는 **합성 샘플 데이터**(4개 계좌 · 48개 종목 · 162건 거래, 2025년 1월 – 2026년 5월)로 돌아갑니다. 실제 18개월 거래 이력은 비공개이고, `DUCKDB_PATH`를 dbt로 빌드한 DB로 지정하면 같은 대시보드를 실데이터로 볼 수 있습니다. 라이브 데모의 수치는 예시일 뿐 실제 성과와는 무관합니다.

**스택:**
```
Fidelity CSV export
    → dbt (DuckDB) staging / intermediate / marts
        → Streamlit + Plotly 대시보드
```

## 페이지 구성

| 페이지 | 내용 |
|---|---|
| Portfolio Overview | 누적 KPI, 계좌별로 쌓은 월별 손익 막대 + 누적 곡선, 분기별 테이블 |
| Ticker Performance | 상위 수익·손실 종목, 승률 vs 손익 버블 차트, 제외 종목, 검색 가능한 전체 테이블 |
| Account Summary | 계좌별 KPI, 누적 성장 곡선, 월별 그룹 막대, 거래 상세 |

## 로컬 실행

```bash
pip install -r requirements.txt

# 전체 데이터 (비공개)
DUCKDB_PATH=/path/to/options_trading.duckdb streamlit run app.py

# 샘플 데이터 (공개)
DUCKDB_PATH=sample_options_trading.duckdb streamlit run app.py
```

## Streamlit Cloud 배포

1. 이 레포를 GitHub에 push (public)
2. [share.streamlit.io](https://share.streamlit.io) 접속
3. 레포 연결 → 메인 파일을 `options-dbt/streamlit_app/app.py`로 지정
4. Secret 추가: `DUCKDB_PATH = "sample_options_trading.duckdb"`
5. 배포 → 포트폴리오용 공개 URL 발급

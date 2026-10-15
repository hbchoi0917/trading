# 옵션 트레이딩 분석 대시보드

[English](README.md) | **한국어**

dbt 분석 파이프라인 위에 Streamlit + DuckDB + Plotly로 만든 대시보드입니다.

**데이터:** 파이프라인은 실제 Fidelity export(비공개)를 처리하도록 만들어졌습니다. 레포에 커밋된
`sample_options_trading.duckdb`는 **완전한 합성 데이터**입니다. 모든 행을 `make_synthetic_sample.py`가
고정 시드로 무작위 생성하고 dbt 마트와 같은 스키마를 쓰기 때문에, 실제 거래·금액·계좌를 노출하지 않고도
앱을 실행하고 배포할 수 있습니다.

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

# 합성 샘플 다시 생성 (고정 시드라 매번 같은 데이터)
python make_synthetic_sample.py sample_options_trading.duckdb
```

## Streamlit Cloud 배포

1. 이 레포를 GitHub에 push (public)
2. [share.streamlit.io](https://share.streamlit.io) 접속
3. 레포 연결 → 메인 파일을 `options-dbt/streamlit_app/app.py`로 지정
4. Secret 추가: `DUCKDB_PATH = "sample_options_trading.duckdb"`
5. 배포 → 포트폴리오용 공개 URL 발급

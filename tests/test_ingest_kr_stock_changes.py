"""KRX 기업 주요 변동사항(상호/업종/액면 변경 이력) 수집·정규화·활용 단위 테스트 (TDD).

pykrx `get_stock_major_changes(ticker)` 는 종목별로 날짜(index)와 상호변경전/후·업종변경전/후·
액면변경전/후·대표이사변경전/후 컬럼을 가진 DataFrame 을 돌려준다(실측 확인). 이 이력을
kr_stock_changes 테이블에 저장하고, 자연어→SQL 질의에서 회사명 매칭이 현재 사명으로 실패할 때
"예전 사명"으로도 종목을 찾을 수 있게 연결한다.

- 파싱: pykrx DataFrame → 정규화 행(없음 표기 "-" 와 액면 0 은 None 으로, 대표이사만 바뀐 행은
  저장 대상 필드가 모두 비므로 스킵 — 스크리닝/백테스트/회사명 매칭에 무관, YAGNI).
- 수집: fetch_fn(DI) 주입으로 네트워크(pykrx) 없이 검증. 실패 종목 격리+Slack 알림, 멱등 upsert.
- 활용: find_stock_code 가 현재 사명 매칭 실패 시 kr_stock_changes(예전 사명)로 폴백해 종목코드를
  찾는다. DB 접근은 임시 SQLite 에 시딩해 사용자 DB 와 완전히 격리한다.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from src.agents.domain_kr import find_stock_code, find_stock_codes
from src.db import connect_readonly, init_db
from src.ingest import kr_stock_changes as ksc


def _conn(db: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


def _major_changes_df(rows: list[dict], dates: list[str]) -> pd.DataFrame:
    """pykrx get_stock_major_changes 출력과 동일한 컬럼/날짜인덱스 DataFrame 을 만든다."""
    cols = [
        "상호변경전", "상호변경후", "업종변경전", "업종변경후",
        "액면변경전", "액면변경후", "대표이사변경전", "대표이사변경후",
    ]
    data = {c: [r.get(c, "-") for r in rows] for c in cols}
    df = pd.DataFrame(data, index=pd.to_datetime(dates))
    df.index.name = "날짜"
    return df


# 삼성전자 실측 형식: 1975 상호신설, 1979 액면분할(1000→500), 1984 상호변경, 2000 대표이사만 변경
_SAMSUNG_DF = _major_changes_df(
    rows=[
        {"상호변경후": "삼성전자공업주", "액면변경전": 0, "액면변경후": 0},
        {"액면변경전": 1000, "액면변경후": 500},
        {"상호변경전": "삼성전자공업주", "상호변경후": "삼성전자주", "액면변경전": 0, "액면변경후": 0},
        {"대표이사변경전": "이건희윤종용", "대표이사변경후": "이건희최도석", "액면변경전": 0, "액면변경후": 0},
    ],
    dates=["1975-06-11", "1979-03-13", "1984-03-23", "2000-01-20"],
)


# --------------------------------------------------------------------------
# 파싱: pykrx DataFrame → 정규화 행
# --------------------------------------------------------------------------
def test_parse_extracts_name_and_par_changes():
    rows = ksc.parse_major_changes(_SAMSUNG_DF, "005930")
    by_date = {r["changed_at"]: r for r in rows}

    # 1975: 상호 신설(변경전 "-"→None, 변경후 값), 액면 0/0→None
    r1975 = by_date["1975-06-11"]
    assert r1975["stock_code"] == "005930"
    assert r1975["name_before"] is None
    assert r1975["name_after"] == "삼성전자공업주"
    assert r1975["par_before"] is None and r1975["par_after"] is None

    # 1979: 액면분할 1000→500 (상호는 "-"→None)
    r1979 = by_date["1979-03-13"]
    assert r1979["name_before"] is None and r1979["name_after"] is None
    assert r1979["par_before"] == 1000
    assert r1979["par_after"] == 500

    # 1984: 상호변경 삼성전자공업주 → 삼성전자주
    r1984 = by_date["1984-03-23"]
    assert r1984["name_before"] == "삼성전자공업주"
    assert r1984["name_after"] == "삼성전자주"


def test_parse_skips_ceo_only_rows():
    # 2000-01-20 은 대표이사만 바뀜(우리가 저장하는 상호/업종/액면은 모두 없음) → 저장 대상 아님
    rows = ksc.parse_major_changes(_SAMSUNG_DF, "005930")
    assert all(r["changed_at"] != "2000-01-20" for r in rows)
    assert len(rows) == 3  # 1975/1979/1984 만 (2000 스킵)


def test_parse_normalizes_sector_changes():
    df = _major_changes_df(
        rows=[{"업종변경전": "제조업", "업종변경후": "서비스업"}],
        dates=["2010-05-01"],
    )
    rows = ksc.parse_major_changes(df, "123456")
    assert len(rows) == 1
    assert rows[0]["sector_before"] == "제조업"
    assert rows[0]["sector_after"] == "서비스업"


def test_parse_empty_dataframe_returns_empty():
    assert ksc.parse_major_changes(pd.DataFrame(), "005930") == []
    assert ksc.parse_major_changes(None, "005930") == []


# --------------------------------------------------------------------------
# 수집: DI fetch_fn 주입(네트워크 없이), 멱등 upsert, 실패 격리
# --------------------------------------------------------------------------
def test_ingest_stores_rows_from_fetch_fn(tmp_path):
    db = str(tmp_path / "s.db")

    def fetch(code: str):
        return ksc.parse_major_changes(_SAMSUNG_DF, code)

    r = ksc.ingest_stock_changes(db_path=db, codes=["005930"], fetch_fn=fetch)
    assert r["tickers"] == 1
    assert r["rows_stored"] == 3
    assert r["failed"] == []

    conn = _conn(db)
    rows = conn.execute(
        "SELECT changed_at, name_before, name_after FROM kr_stock_changes "
        "WHERE stock_code='005930' ORDER BY changed_at").fetchall()
    assert [row["changed_at"] for row in rows] == ["1975-06-11", "1979-03-13", "1984-03-23"]
    conn.close()


def test_ingest_is_idempotent_same_run(tmp_path):
    db = str(tmp_path / "idem.db")

    def fetch(code: str):
        return ksc.parse_major_changes(_SAMSUNG_DF, code)

    ksc.ingest_stock_changes(db_path=db, codes=["005930"], fetch_fn=fetch)
    ksc.ingest_stock_changes(db_path=db, codes=["005930"], fetch_fn=fetch)

    conn = _conn(db)
    total = conn.execute(
        "SELECT COUNT(*) FROM kr_stock_changes WHERE stock_code='005930'").fetchone()[0]
    assert total == 3  # 재실행해도 (종목,날짜) UNIQUE 로 중복 없음
    conn.close()


def test_ingest_defaults_to_company_table_when_codes_none(tmp_path):
    db = str(tmp_path / "auto.db")
    init_db(db)
    conn = _conn(db)
    conn.execute("INSERT INTO company(stock_code, name, market, sector) VALUES('005930','삼성전자','KOSPI','반도체')")
    conn.execute("INSERT INTO company(stock_code, name, market, sector) VALUES('000660','SK하이닉스','KOSPI','반도체')")
    conn.commit()
    conn.close()

    seen: list[str] = []

    def fetch(code: str):
        seen.append(code)
        return []

    r = ksc.ingest_stock_changes(db_path=db, fetch_fn=fetch)
    assert set(seen) == {"005930", "000660"}
    assert r["tickers"] == 2


def test_ingest_isolates_failed_ticker_and_continues(tmp_path, monkeypatch):
    db = str(tmp_path / "fail.db")
    alerts: list[str] = []
    monkeypatch.setattr(ksc, "send_slack_alert", lambda msg: alerts.append(msg) or True)

    def fetch(code: str):
        if code == "BAD":
            raise RuntimeError("pykrx 조회 실패(mock)")
        return ksc.parse_major_changes(_SAMSUNG_DF, code)

    r = ksc.ingest_stock_changes(db_path=db, codes=["BAD", "005930"], fetch_fn=fetch)
    assert r["failed"] == ["BAD"]
    assert r["tickers"] == 1          # 성공한 종목만 카운트
    assert r["rows_stored"] == 3      # 실패 종목은 격리, 나머지는 정상 저장
    assert len(alerts) == 1           # 실패 종목당 Slack 알림 1회

    conn = _conn(db)
    ok = conn.execute("SELECT COUNT(*) FROM kr_stock_changes WHERE stock_code='005930'").fetchone()[0]
    assert ok == 3
    conn.close()


def test_ingest_uses_di_and_never_calls_pykrx(tmp_path):
    # fetch_fn 주입 시 실제 pykrx 호출을 절대 하지 않는다(네트워크 없이 동작).
    db = str(tmp_path / "di.db")
    called = {"n": 0}

    def fetch(code: str):
        called["n"] += 1
        return []

    ksc.ingest_stock_changes(db_path=db, codes=["005930", "000660"], fetch_fn=fetch)
    assert called["n"] == 2


# --------------------------------------------------------------------------
# 활용(핵심): 예전 사명으로 물어봐도 현재 종목코드를 찾는다
# --------------------------------------------------------------------------
def _seed_name_history(db: str) -> None:
    """company(현재 사명) + kr_stock_changes(예전 사명 이력)를 시딩한다."""
    init_db(db)
    conn = _conn(db)
    # 현재 사명만 company 에 있음
    conn.execute("INSERT INTO company(stock_code, name, market, sector) VALUES('111111','새이름전자','KOSPI','반도체')")
    conn.execute("INSERT INTO company(stock_code, name, market, sector) VALUES('005930','삼성전자','KOSPI','반도체')")
    # 예전 사명 이력: 옛이름전자 → 중간이름전자 → 새이름전자
    conn.execute(
        "INSERT INTO kr_stock_changes(stock_code, changed_at, name_before, name_after) "
        "VALUES('111111','2010-01-01','옛이름전자','중간이름전자')")
    conn.execute(
        "INSERT INTO kr_stock_changes(stock_code, changed_at, name_before, name_after) "
        "VALUES('111111','2015-01-01','중간이름전자','새이름전자')")
    conn.commit()
    conn.close()


def test_find_stock_code_falls_back_to_former_name(tmp_path):
    db = str(tmp_path / "hist.db")
    _seed_name_history(db)
    conn = connect_readonly(db)
    try:
        # 현재 사명(company)에 없는 예전 사명으로 물어봐도 종목코드를 찾는다
        assert find_stock_code(conn, "옛이름전자 실적 알려줘") == "111111"
    finally:
        conn.close()


def test_find_stock_code_matches_intermediate_former_name(tmp_path):
    db = str(tmp_path / "hist2.db")
    _seed_name_history(db)
    conn = connect_readonly(db)
    try:
        # 중간이름(name_after/name_before 양쪽에 등장)으로도 찾는다
        assert find_stock_code(conn, "중간이름전자 주가") == "111111"
    finally:
        conn.close()


def test_find_stock_code_prefers_current_name_over_former(tmp_path):
    db = str(tmp_path / "hist3.db")
    _seed_name_history(db)
    conn = connect_readonly(db)
    try:
        # 현재 사명이 매칭되면 예전 사명 폴백은 쓰지 않는다(회귀 없음)
        assert find_stock_code(conn, "삼성전자 PER") == "005930"
    finally:
        conn.close()


def test_find_stock_code_returns_none_when_neither_current_nor_former(tmp_path):
    db = str(tmp_path / "hist4.db")
    _seed_name_history(db)
    conn = connect_readonly(db)
    try:
        assert find_stock_code(conn, "존재하지않는회사 실적") is None
    finally:
        conn.close()


def test_find_stock_codes_includes_former_name_match(tmp_path):
    db = str(tmp_path / "hist5.db")
    _seed_name_history(db)
    conn = connect_readonly(db)
    try:
        codes = find_stock_codes(conn, "옛이름전자와 삼성전자 비교")
        assert "111111" in codes  # 예전 사명 폴백
        assert "005930" in codes  # 현재 사명
    finally:
        conn.close()


def test_former_name_fallback_safe_when_table_missing(tmp_path):
    # kr_stock_changes 가 비어도(이력 없음) 현재 사명 매칭은 정상 동작하고 크래시 없음
    db = str(tmp_path / "empty_hist.db")
    init_db(db)
    conn = _conn(db)
    conn.execute("INSERT INTO company(stock_code, name, market, sector) VALUES('005930','삼성전자','KOSPI','반도체')")
    conn.commit()
    conn.close()
    ro = connect_readonly(db)
    try:
        assert find_stock_code(ro, "삼성전자 PER") == "005930"
        assert find_stock_code(ro, "옛이름전자 실적") is None
    finally:
        ro.close()

"""가격 데이터 품질 게이트 — 가격 불연속으로 오염된 (종목, 시점) 조합을 계산에서 제외한다.

배경
----
prices.close 에 "하루 만에 종가가 2배 이상 뛰거나 절반 이하로 떨어지는" 구간을 가진
종목이 있다. 원인은 대부분 액면분할/병합이 수정주가로 소급 반영되지 않은 데이터
버그이거나(예: 저유동 동전주가 오랫동안 동일 종가로 고정됐다가 정수배로 점프),
드물게는 원본 파싱 오류다(예: 종가 1원이 몇 달 이어지다 67000원으로 점프).

원인이 다양하고 계속 새로운 예외 케이스가 발견되므로(예: 246830 시냅스엠은 상승
정수배 점프 외에 하락 폭락 구간도 섞여 있어 배율 하나로 "복원"이 안 됐고, 상승만
보던 기존 임계값(abs(ratio-1)>1.0, scripts/fix_split_discontinuities.py)은 하락
폭락을 아예 못 잡는 사각지대였다), 배율을 찾아 곱해서 "고치는" 접근 대신 **오염된
가격 구간을 계산에서 빼는** 더 안전하고 유지보수하기 쉬운 접근을 쓴다.

판정
----
인접 거래일 종가비가 2배 이상(ratio>=2.0) 이거나 1/2 이하(ratio<=0.5)면 그 날을
'이상 발생일 D'로 본다(상승/하락 대칭 — 배율로 보면 2배 점프든 1/2배 급락이든 크기가
같은 비정상 변동이다). 제외는 **종목 전체를 영구히 빼는 게 아니라 D 근처의 백테스트
시점(asof)에서만** 한다(asof-aware 국소 제외 — 윈도우 근거는 PRICE_ANOMALY_WINDOW_*
상수와 get_price_quality_excluded_codes 참조). 종목 전체를 모든 시점에서 빼면, 상폐
직전 정리매매/감자/거래재개 등으로 큰 변동을 겪은 종목의 '정상 거래기간' 데이터까지
함께 사라져(상폐 종목 41%가 통째로 제외됐다) 생존편향이 다른 경로로 재유입된다 — 이
게이트가 바로 그 편향을 막으려는 것이므로 국소 제외로 좁혔다. 단, 진짜 데이터오류를
잡는 안전장치 자체는 유지한다(정리매매/감자 여부를 과거 이력에서 판별할 데이터소스가
없어 '진짜 이상치 vs 합법적 변동'은 구분하지 않는다 — 범위만 좁힌다).

캐싱
----
전체 스캔(prices ~840만 행, LAG 윈도우)은 실측 약 6~7초 걸린다(2026-07 data/market.db
기준). 백테스트 1회 실행이 리밸런싱 시점마다 유니버스(=metrics_at)를 여러 번 구성하는
구조라 요청마다 돌리기엔 비싸므로, ingest_meta에 '종목→이상 발생일 목록' 매핑을 JSON으로
캐싱한다(기존 usdkrw_rate/inactive_codes 캐싱 관례와 동일 패턴 — src/ingest/dart.py
inactive_codes, src/ingest/exchange_rate.py usdkrw_rate 참고). 최초 호출 시 스캔해 저장하고,
이후 호출은 ingest_meta 단건 조회(수 ms) + asof별 윈도우 필터(메모리 내)로 끝난다.

캐시가 있으면 재스캔하지 않으므로, 새로 적재된 가격에서 생긴 이상치는 즉시 반영되지
않는다(다음 refresh=True 호출 전까지). 이 트레이드오프는 의도적이다 — 하루이틀 늦게
걸러져도(최악의 경우) 매 요청마다 6~7초를 태우는 것보다 낫다.
"""
from __future__ import annotations

import json
import sqlite3

from .db import get_meta, set_meta
from .version import now_iso

# fix_split_discontinuities.py 의 상승전용 임계값(JUMP_THRESHOLD=1.0, 즉 ratio>2.0)을
# 하락 방향(ratio<=0.5)까지 대칭으로 확장했다. 정수배 여부는 따지지 않는다 — 이 모듈은
# "정수배로 보정 가능한가"가 아니라 "이 종목의 과거 이력을 신뢰할 수 있는가"만 판단한다.
JUMP_RATIO_HIGH = 2.0
JUMP_RATIO_LOW = 0.5

# ── asof-aware 국소 제외 윈도우 ──────────────────────────────────────────────
# 이상치(가격 불연속)는 특정 거래일 D 에 발생한다. 백테스트 지표는 모두 과거만 보므로
# (룩어헤드 금지) D 이전 asof 는 원래 오염되지 않지만, 다일에 걸친 이벤트/경계 흔들림에
# 대한 안전여유로 D 이전 1개월까지만 제외한다. D 이후는 return_12m(12개월 룩백) 수익률
# 윈도우 [asof-12m, asof] 가 D 를 품는 동안(asof∈[D, D+12m]) 수익률이 오염되므로 12개월을
# 제외한다. 그 밖의(충분히 먼) asof 에서는 그 종목의 정상 거래기간 데이터를 살려둔다 —
# 상폐 직전 정리매매/감자 등의 큰 변동 때문에 종목이 통째로 빠져 생존편향이 재유입되던
# 문제를 막는다(설계 근거는 get_price_quality_excluded_codes 참조).
PRICE_ANOMALY_WINDOW_BEFORE_MONTHS = 1
PRICE_ANOMALY_WINDOW_AFTER_MONTHS = 12

# 캐시는 '종목→이상 발생일 목록' 매핑(JSON). 예전엔 종목코드 CSV 만 캐싱했으나, 국소
# 제외가 발생일을 알아야 하므로 발생일까지 담는 새 키로 바꿨다(구 CSV 캐시와 형식이
# 달라 혼선을 막으려 키 이름도 교체 — 구 캐시는 자연히 무시되고 최초 호출 때 재계산된다).
_CACHE_KEY_DATES = "price_quality_anomaly_dates"
_CACHE_KEY_AT = "price_quality_excluded_at"

# ── 물적분할(스핀오프) 시차 가드 임계값 ──────────────────────────────────────
# 물적분할 직후엔 주가(=시총)가 시장에서 즉시 급락 반영되지만, 자기자본(재무제표)은 다음 분기
# DART 공시가 나와야 갱신된다. 그 시차 구간엔 effective_quarter_at 이 '분할 전(자회사 포함) 큰
# 자기자본'을 그대로 쓰므로 PBR = 시총 ÷ 자기자본 이 비정상적으로 낮게 계산돼 저평가 오탐이 난다.
# 판정 지표는 시총/지배주주지분 비율(= 사실상 PBR)이고, 이 비율이 직전 유효분기(≈3개월 전) 대비
# 얼마나 급변했는지를 본다. 임계값 2.0/0.5 는 위 가격이상치 가드(JUMP_RATIO_HIGH/LOW)와 동일한
# '배율로 대칭' 감각을 그대로 가져왔다 — 물적분할된 모회사 시총은 흔히 30~50%+ 급락하므로 비율이
# 0.5배(반토막) 이하로 떨어지면 그 시차 오탐을 잡아낸다(급등 방향 2.0배는 역합병 등 대칭 케이스).
# 이 임계값은 '간이 방식'이라 genuine 한 ±50% 분기 급등락도 함께 걸릴 수 있으나, (1) 종목 전체가
# 아니라 그 asof 의 밸류에이션 배수만 결측이고 (2) 다음 분기 공시로 자연 정상화되며 (3) 반토막 난
# 종목의 PBR 은 어차피 신뢰도가 낮으므로 허용 가능한 트레이드오프다(과도한 정교화를 피한다).
SPINOFF_RATIO_HIGH = 2.0
SPINOFF_RATIO_LOW = 0.5
SPINOFF_LOOKBACK_MONTHS = 3  # 직전 유효분기 ≈ 3개월 전


def detect_price_quality_anomaly_dates(
    conn: sqlite3.Connection,
    ratio_high: float = JUMP_RATIO_HIGH,
    ratio_low: float = JUMP_RATIO_LOW,
) -> dict[str, list[str]]:
    """전체 이력을 스캔해, 인접 거래일 종가비가 [ratio_low, ratio_high] 구간 밖인 '이상 발생일'을
    종목별로 모은 매핑({stock_code: [date, ...]}).

    date 는 이상 변동이 나타난 거래일(비율의 분자 쪽 — 점프/폭락이 관측된 날)이다. asof-aware
    국소 제외(get_price_quality_excluded_codes)가 이 발생일을 기준으로 제외 윈도우를 잡는다.
    scripts/fix_split_discontinuities._raw_jumps 와 동일한 LAG 윈도우 SQL을 대칭 임계값으로
    재사용한다. 캐시를 거치지 않는 순수 스캔 함수라 실제 대용량 DB에서는 수 초가 걸릴 수 있다
    (캐싱은 get_price_quality_anomaly_dates가 담당). 종목별 date 는 오름차순으로 정렬된다.
    """
    rows = conn.execute(
        """
        WITH ch AS (
          SELECT stock_code, date, close,
                 LAG(close) OVER (PARTITION BY stock_code ORDER BY date) AS prev_close
          FROM prices WHERE close IS NOT NULL AND close > 0
        )
        SELECT stock_code, date FROM ch
        WHERE prev_close IS NOT NULL AND prev_close > 0
          AND (close / prev_close >= ? OR close / prev_close <= ?)
        ORDER BY stock_code, date
        """,
        (ratio_high, ratio_low),
    ).fetchall()
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r["stock_code"], []).append(r["date"])
    return out


def detect_price_quality_anomalies(
    conn: sqlite3.Connection,
    ratio_high: float = JUMP_RATIO_HIGH,
    ratio_low: float = JUMP_RATIO_LOW,
) -> set[str]:
    """이상치가 하나라도 있는 종목코드 집합(발생일 정보를 버린 얇은 래퍼).

    detect_price_quality_anomaly_dates 의 키 집합이다 — '이 종목의 과거 이력에 불연속이
    있는가'만 알면 되는 진단·통계용 하위호환 API. 실제 백테스트 제외는 asof-aware 국소
    판정(get_price_quality_excluded_codes)을 쓴다.
    """
    return set(detect_price_quality_anomaly_dates(conn, ratio_high, ratio_low).keys())


def get_price_quality_anomaly_dates(
    conn: sqlite3.Connection, refresh: bool = False
) -> dict[str, list[str]]:
    """종목별 가격 이상 발생일 매핑(ingest_meta JSON 캐시 우선).

    refresh=False(기본): 캐시가 있으면 그대로 반환(재스캔 없음). 캐시가 없으면(최초
    호출) 자동으로 스캔해 채운다.
    refresh=True: 캐시 유무와 무관하게 강제로 재스캔하고 캐시를 갱신한다(예: 일일
    가격 적재 후 최신화하려는 배치 호출용).
    """
    if not refresh:
        cached = get_meta(conn, _CACHE_KEY_DATES)
        if cached is not None:
            return json.loads(cached)

    dates_map = detect_price_quality_anomaly_dates(conn)
    try:
        set_meta(conn, _CACHE_KEY_DATES, json.dumps(dates_map, sort_keys=True))
        set_meta(conn, _CACHE_KEY_AT, now_iso())
        conn.commit()
    except sqlite3.OperationalError:
        # 읽기전용 연결(connect_readonly — LLM SQL 실행/에이전트 응답 경로가 방어적으로
        # 쓰는 연결)에서는 캐시에 쓸 수 없다("attempt to write a readonly database").
        # 계산 결과 자체는 정상 반환하고 캐시 저장만 건너뛴다 — 다음에 쓰기가능
        # 연결(배치 갱신 스크립트 등)이 호출될 때 캐시가 채워진다.
        pass
    return dates_map


def get_price_quality_excluded_codes(
    conn: sqlite3.Connection, asof: str | None = None, refresh: bool = False
) -> set[str]:
    """제외 대상 종목코드 집합.

    asof 를 주면 국소(asof-aware) 제외 — 각 종목의 이상 발생일 D 주변
    [D-BEFORE, D+AFTER] 윈도우(PRICE_ANOMALY_WINDOW_*_MONTHS)에 asof 가 들어갈 때만 그
    종목을 제외한다. 이상치와 충분히 먼(과거) asof 에서는 제외하지 않아, 상폐 직전 정리매매
    /감자 등으로 큰 변동을 겪은 종목의 정상 거래기간 데이터가 모든 시점에서 함께 사라져
    생존편향이 재유입되던 문제를 막는다.

    asof 를 생략(None)하면 하위호환으로 '이상치가 하나라도 있는 모든 종목'을 반환한다
    (진단·통계 등 asof 문맥이 없는 호출용).
    """
    dates_map = get_price_quality_anomaly_dates(conn, refresh=refresh)
    if asof is None:
        return set(dates_map.keys())

    # D±개월 경계는 backtest.data_access._months_before 로 잡는다(모듈 순환참조를 피하려
    # 함수 안에서 지연 import — is_equity_ratio_anomalous 와 동일 패턴). 음수 개월은 미래
    # 방향(D+after)을 뜻한다. ISO 날짜는 사전식 비교 = 시간 비교라 문자열로 바로 대소 비교한다.
    from .backtest.data_access import _months_before

    excluded: set[str] = set()
    for code, dates in dates_map.items():
        for d in dates:
            lo = _months_before(d, PRICE_ANOMALY_WINDOW_BEFORE_MONTHS)   # D - before
            hi = _months_before(d, -PRICE_ANOMALY_WINDOW_AFTER_MONTHS)   # D + after
            if lo <= asof <= hi:
                excluded.add(code)
                break
    return excluded


def is_equity_ratio_anomalous(
    conn: sqlite3.Connection,
    stock_code: str,
    asof: str,
    *,
    ratio_high: float = SPINOFF_RATIO_HIGH,
    ratio_low: float = SPINOFF_RATIO_LOW,
    lookback_months: int = SPINOFF_LOOKBACK_MONTHS,
) -> bool:
    """물적분할(스핀오프) 시차로 PBR/PSR 류가 오염됐는지 판정하는 '일시적' 가드.

    블랭킷 가격이상치 가드(detect_price_quality_anomalies, 종목 전체 영구 제외)와 성격이
    다르다. 물적분할 오탐은 일시적이다 — 다음 분기 재무제표가 공시되면 자기자본이 갱신돼
    자연 정상화된다. 그래서 이 함수는 종목을 빼는 게 아니라, 특정 (종목, asof) 조합에서만
    True 를 돌려주고(호출부가 그 시점의 밸류에이션 배수만 결측 처리하게 함) 별도 해제 로직 없이
    다음 분기엔 자연히 False 로 돌아온다.

    판정: 시총 ÷ 지배주주지분 비율(≈PBR)을 asof 와 lookback_months 전(≈직전 유효분기) 두
    시점에서 각각 구해, 그 변화율이 ratio_low 이하(급락)이거나 ratio_high 이상(급등)이면 True.
    각 시점의 유효분기·자기자본·시총은 metrics_at 과 동일한 look-ahead 규약(effective_quarter_at
    /controlling_equity(asof=)/date<=asof 최근 종가)으로 조회하므로 미래참조가 없다.

    판정에 필요한 값(현재/기준 시점의 유효분기·자기자본·시총) 중 하나라도 없으면 판단하지 않고
    안전하게 False 를 반환한다(결측 처리하지 않음 — 근거가 있을 때만 가린다).

    backtest.data_access ↔ data_quality 의 모듈 순환참조를 피하려고 필요한 헬퍼는 함수 안에서
    지연 import 한다(backtest.data_access 는 상단에서 이 모듈을 import 한다).
    """
    from .backtest.data_access import _months_before, _price_at, effective_quarter_at
    from .ingest.metrics import controlling_equity

    def _cap_equity_ratio(at: str) -> float | None:
        q = effective_quarter_at(conn, stock_code, at)
        if not q:
            return None
        equity = controlling_equity(conn, stock_code, q, asof=at)
        _, cap = _price_at(conn, stock_code, at)
        if not (equity and equity > 0 and cap and cap > 0):
            return None
        return cap / equity

    ratio_now = _cap_equity_ratio(asof)
    if ratio_now is None:
        return False
    ratio_ref = _cap_equity_ratio(_months_before(asof, lookback_months))
    if not ratio_ref:  # None 또는 0 → 기준이 없어 판단 불가
        return False
    change = ratio_now / ratio_ref
    return change >= ratio_high or change <= ratio_low

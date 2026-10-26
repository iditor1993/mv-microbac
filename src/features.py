"""两库统一的特征构造：ARMD-MGB 与 MIMIC-IV 用完全相同的定义。

本研究（MV_microbac）相对 C_prior_history_value 的改动
-----------------------------------------------------
* 队列：下呼吸道标本 + 革兰阴性菌 + 头孢吡肟 S/R + **机械通气**（ARMD 用开机日窗口代理，
  MIMIC 用有创通气区间重叠）。
* 目标菌不再单一：用 org_pa / org_other 两个基准协变量吸收病例组合。
* HIST 由二值改为**三层编码**（无 / 有同菌 / 有同菌药敏全 S / 有同菌药敏含 R）。
* 新增 d_cefepime（结局药物本身的既往暴露）。
* 新增同菌**异部位**特征（呼吸道 vs 非呼吸道），供 S02 使用。

特征组
------
  DEMO     人口学
  ORG      本次菌种组合（M0 起即入模型）
  ABX      既往抗菌药暴露（药名正则，两库共用）
  CXBURDEN 既往培养数量与时间
  HIST_SAME  本次菌种的既往史（三层）
  HIST_OTHER 其他革兰阴性菌的既往史

设计约束
--------
* 只用药名正则，不用数据库自带药物类别（两库类别口径不一致）。
* 无既往暴露用固定 sentinel 9999，不做数据驱动插补。
* 一律严格 prior_time < index_time，同日不算既往。
* 既往窗口对“同菌”与“同菌药敏”统一使用同一窗口，避免 C 项目 P6b 记录的不对称。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SENTINEL = 9999.0

DEMO = ["age_mid", "male"]
ORG = ["org_pa", "org_other"]
ABX = [
    "days_since_abx", "n_drugs_365d",
    "d_fq", "d_bsc", "d_carbapenem", "d_piptazo",
    "d_tmpsmx", "d_nitro", "d_amino", "d_glyco", "d_cefepime",
]
CXBURDEN = ["n_cx_90d", "n_cx_365d", "days_since_cx"]
HIST_SAME = ["prior_same_org", "prior_same_ast", "prior_same_res"]
HIST_OTHER = ["prior_other_gn_org", "prior_other_gn_res"]

# 两库通用的革兰阴性菌正则（不含 HAEMOPHILUS，见 protocol D11）
GN_PATTERN = ("PSEUDOMONAS|KLEBSIELLA|ESCHERICHIA|ENTEROBACTER|SERRATIA|PROTEUS"
              "|CITROBACTER|ACINETOBACTER|STENOTROPHOMONAS")
GN_OTHER_PATTERN = "ACINETOBACTER|STENOTROPHOMONAS"

_NORM = r"trim(regexp_replace(upper({col}), '\s*\(.*\)$', ''))"

DRUGS = {
    "d_fq": "ciprofloxacin|levofloxacin|moxifloxacin",
    "d_bsc": "ceftriaxone|cefotaxime|ceftazidime",
    "d_cefepime": "cefepime",
    "d_carbapenem": "meropenem|ertapenem|imipenem|doripenem",
    "d_piptazo": "piperacillin",
    "d_tmpsmx": "sulfameth",
    "d_nitro": "nitrofurantoin",
    "d_amino": "gentamicin|tobramycin|amikacin",
    "d_glyco": "vancomycin",
}

AGE_BANDS = [
    (18, 25, 21.0), (25, 35, 29.5), (35, 45, 39.5), (45, 55, 49.5),
    (55, 65, 59.5), (65, 75, 69.5), (75, 85, 79.5), (85, 90, 87.0), (90, 300, 92.0),
]
ARMD_AGE_MID = {
    "18-24 years": 21.0, "25-34 years": 29.5, "35-44 years": 39.5,
    "45-54 years": 49.5, "55-64 years": 59.5, "65-74 years": 69.5,
    "75-84 years": 79.5, "85-89 years": 87.0, "above 90": 92.0,
}


def band_age(age: float) -> float:
    for lo, hi, mid in AGE_BANDS:
        if lo <= age < hi:
            return mid
    return np.nan


def _scope_sql(scope: str, col: str) -> str:
    """既往标本范围：all / resp / nonresp。"""
    if scope == "all":
        return "TRUE"
    if scope == "resp":
        return f"{col} = 'RESPIRATORY_TRACT'"
    if scope == "nonresp":
        return f"{col} <> 'RESPIRATORY_TRACT'"
    raise ValueError(scope)


# ---------------------------------------------------------------------------
# ARMD-MGB
# ---------------------------------------------------------------------------
def build_armd(con, mv_days: int = 7, history_days: int = 365,
               scope: str = "all", drug: str = "cefepime") -> pd.DataFrame:
    """ARMD：通气状态用“开机日距培养 0–mv_days 天”代理（无拔管时间）。"""
    scope_cond = _scope_sql(scope, "p.culture_description")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE mv_ep AS
        SELECT DISTINCT anon_id,
               cast(order_time_jittered_utc_shifted AS DATE)
                 - to_days(procedure_days_culture) AS mv_date
        FROM main.prior_procedures
        WHERE procedure_description = 'mechvent' AND procedure_days_culture IS NOT NULL
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE idx0 AS
        SELECT order_proc_id_coded AS cid, anon_id,
               min(order_time_jittered_utc_shifted) AS d,
               max(CASE WHEN CLSI_2022_pheno = 'Resistant' THEN 1 ELSE 0 END) AS y
        FROM main.microbiology_cohort
        WHERE culture_description = 'RESPIRATORY_TRACT'
          AND antibiotic = '{drug}'
          AND CLSI_2022_pheno IN ('Resistant', 'Susceptible')
          AND regexp_matches(organism, '{GN_PATTERN}')
        GROUP BY 1, 2
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE idx AS
        SELECT i.cid, i.anon_id, i.d, i.y,
               min(datediff('day', e.mv_date, i.d)) FILTER (WHERE e.mv_date <= i.d) AS mv_days
        FROM idx0 i
        LEFT JOIN mv_ep e ON e.anon_id = i.anon_id
             AND e.mv_date <= i.d AND e.mv_date >= i.d - INTERVAL 365 DAY
        GROUP BY 1, 2, 3, 4
    """)
    con.execute(f"CREATE OR REPLACE TEMP TABLE cohort AS SELECT * FROM idx WHERE mv_days <= {int(mv_days)}")

    # 本次菌种（新生物种集合）
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE idx_org_long AS
        SELECT DISTINCT c.cid, {_NORM.format(col='m.organism')} AS org_norm,
               regexp_matches(upper(m.organism), '{GN_OTHER_PATTERN}') AS is_other_gn,
               regexp_matches(upper(m.organism), 'PSEUDOMONAS') AS is_pa,
               upper(m.organism) AS organism_raw
        FROM cohort c
        JOIN main.microbiology_cohort m ON m.order_proc_id_coded = c.cid
        WHERE m.organism IS NOT NULL AND regexp_matches(m.organism, '{GN_PATTERN}')
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE idx_org AS
        SELECT cid,
               max(is_pa::int)      AS org_pa,
               max(is_other_gn::int) AS org_other,
               count(DISTINCT org_norm) AS n_org
        FROM idx_org_long GROUP BY 1
    """)

    # 既往微生物（窗口内、范围过滤后）
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE prior AS
        SELECT c.cid AS icid,
               {_NORM.format(col='p.organism')} AS org_norm,
               regexp_matches(upper(p.organism), '{GN_PATTERN}') AS is_gn,
               p.antibiotic AS ab,
               p.CLSI_2022_pheno AS ph,
               (p.culture_description = 'RESPIRATORY_TRACT') AS is_resp
        FROM cohort c
        JOIN main.microbiology_cohort p
          ON p.anon_id = c.anon_id
         AND p.organism IS NOT NULL
         AND p.order_time_jittered_utc_shifted < c.d
         AND p.order_time_jittered_utc_shifted >= c.d - INTERVAL {int(history_days)} DAY
        WHERE {scope_cond}
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE prior_flag AS
        SELECT pr.*, (io.org_norm IS NOT NULL) AS is_same
        FROM prior pr
        LEFT JOIN idx_org_long io ON io.cid = pr.icid AND io.org_norm = pr.org_norm
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE f_hist AS
        SELECT c.cid,
               coalesce(max(CASE WHEN f.is_same THEN 1 ELSE 0 END), 0) AS prior_same_org,
               coalesce(max(CASE WHEN f.is_same AND f.ab = '{drug}'
                                  AND f.ph IN ('Resistant', 'Susceptible') THEN 1 ELSE 0 END), 0) AS prior_same_ast,
               coalesce(max(CASE WHEN f.is_same AND f.ab = '{drug}'
                                  AND f.ph = 'Resistant' THEN 1 ELSE 0 END), 0) AS prior_same_res,
               coalesce(max(CASE WHEN f.is_same AND f.ab = '{drug}'
                                  AND f.ph = 'Resistant' AND f.is_resp THEN 1 ELSE 0 END), 0) AS prior_same_res_resp,
               coalesce(max(CASE WHEN f.is_same AND f.ab = '{drug}'
                                  AND f.ph = 'Resistant' AND NOT f.is_resp THEN 1 ELSE 0 END), 0) AS prior_same_res_nonresp,
               coalesce(max(CASE WHEN f.is_gn AND NOT f.is_same THEN 1 ELSE 0 END), 0) AS prior_other_gn_org,
               coalesce(max(CASE WHEN f.is_gn AND NOT f.is_same AND f.ab = '{drug}'
                                  AND f.ph = 'Resistant' THEN 1 ELSE 0 END), 0) AS prior_other_gn_res
        FROM cohort c LEFT JOIN prior_flag f ON f.icid = c.cid
        GROUP BY 1
    """)

    # 培养负担（全部标本类型）
    con.execute("""
        CREATE OR REPLACE TEMP TABLE cx_all AS
        SELECT DISTINCT anon_id, order_proc_id_coded AS cid,
               order_time_jittered_utc_shifted AS cd
        FROM main.microbiology_cohort
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE f_cx AS
        SELECT c.cid,
               count(DISTINCT p.cid) FILTER (WHERE p.cd < c.d AND p.cd >= c.d - INTERVAL 90 DAY)  AS n_cx_90d,
               count(DISTINCT p.cid) FILTER (WHERE p.cd < c.d AND p.cd >= c.d - INTERVAL 365 DAY) AS n_cx_365d,
               coalesce(datediff('day', max(p.cd) FILTER (WHERE p.cd < c.d), c.d), 9999) AS days_since_cx
        FROM cohort c LEFT JOIN cx_all p ON p.anon_id = c.anon_id
        GROUP BY 1, c.d
    """)

    # 既往抗菌药
    flags = ",\n".join(
        f"""coalesce(max(CASE WHEN regexp_matches(lower(a.medication_name), '{pat}') THEN 1 ELSE 0 END)
                     FILTER (WHERE a.last_dose_to_culture > 0
                             AND a.last_dose_to_culture <= 365), 0) AS {name}"""
        for name, pat in DRUGS.items()
    )
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE f_abx AS
        SELECT c.cid,
               coalesce(min(a.last_dose_to_culture) FILTER (WHERE a.last_dose_to_culture > 0), 9999) AS days_since_abx,
               coalesce(min(a.last_dose_to_culture) FILTER (WHERE a.last_dose_to_culture > 0
                     AND regexp_matches(lower(a.medication_name), 'cefepime')), 9999) AS days_since_cefepime,
               count(DISTINCT a.medication_name)
                     FILTER (WHERE a.last_dose_to_culture > 0
                             AND a.last_dose_to_culture <= 365) AS n_drugs_365d,
               {flags}
        FROM cohort c
        LEFT JOIN main.prior_abx a
          ON a.anon_id = c.anon_id AND a.order_proc_id_coded = c.cid
        GROUP BY 1
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE demo AS
        SELECT anon_id, order_proc_id_coded, any_value(age) AS age, any_value(gender) AS gender
        FROM main.demographics GROUP BY 1, 2
    """)

    df = con.execute("""
        SELECT c.anon_id AS patient_id, c.cid, c.y, c.d AS index_date, c.mv_days,
               d.age, d.gender,
               t.org_pa, t.org_other, t.n_org,
               x.n_cx_90d, x.n_cx_365d, x.days_since_cx,
               h.prior_same_org, h.prior_same_ast, h.prior_same_res,
               h.prior_same_res_resp, h.prior_same_res_nonresp,
               h.prior_other_gn_org, h.prior_other_gn_res,
               a.days_since_abx, a.days_since_cefepime, a.n_drugs_365d,
               a.d_fq, a.d_bsc, a.d_cefepime, a.d_carbapenem, a.d_piptazo,
               a.d_tmpsmx, a.d_nitro, a.d_amino, a.d_glyco
        FROM cohort c
        LEFT JOIN f_hist h  ON h.cid = c.cid
        LEFT JOIN f_cx   x  ON x.cid = c.cid
        LEFT JOIN f_abx  a  ON a.cid = c.cid
        LEFT JOIN idx_org t ON t.cid = c.cid
        LEFT JOIN demo   d  ON d.anon_id = c.anon_id AND d.order_proc_id_coded = c.cid
        ORDER BY c.anon_id, c.d, c.cid
    """).df()

    df["age_mid"] = df["age"].map(ARMD_AGE_MID)
    df["male"] = (df["gender"] == "Male").astype(int)
    df["system"] = "ARMD-MGB"
    df["year"] = np.nan
    return df


# ---------------------------------------------------------------------------
# MIMIC-IV
# ---------------------------------------------------------------------------
RESP_SPECS = ("'SPUTUM','BRONCHOALVEOLAR LAVAGE','BRONCHIAL WASHINGS','Mini-BAL',"
              "'TRACHEAL ASPIRATE','BRONCHIAL BRUSH','BRONCHIAL BRUSH - PROTECTED'")


def build_mimic(con, tol_hours: int = 12, history_days: int = 365,
                scope: str = "all", drug: str = "CEFEPIME",
                imv_filter: bool = True) -> pd.DataFrame:
    """MIMIC：通气状态 = 有创通气/气切区间与培养时点重叠（±tol_hours）。

    imv_filter=False 时不限制通气，用于 S05 表型模拟（把 ARMD 的"开机日窗口"规则
    原样套到 MIMIC，再与真实通气重叠比较）。
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE idx0 AS
        SELECT micro_specimen_id AS cid,
               any_value(subject_id) AS subject_id,
               any_value(hadm_id)    AS hadm_id,
               min(chartdate)        AS d,
               max(CASE WHEN interpretation = 'R' THEN 1 ELSE 0 END) AS y
        FROM hosp.microbiologyevents
        WHERE spec_type_desc IN ({RESP_SPECS})
          AND org_name IS NOT NULL
          AND regexp_matches(org_name, '{GN_PATTERN}')
          AND ab_name = '{drug}' AND interpretation IN ('S', 'R')
        GROUP BY 1
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE mv AS
        SELECT s.subject_id, v.starttime, v.endtime
        FROM derived.ventilation v
        JOIN (SELECT subject_id, stay_id FROM icu.icustays) s USING (stay_id)
        WHERE v.ventilation_status IN ('InvasiveVent', 'Tracheostomy')
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE idx AS
        SELECT i.*,
               max(CASE WHEN m.starttime <= i.d + INTERVAL {int(tol_hours)} HOUR
                         AND m.endtime  >= i.d - INTERVAL {int(tol_hours)} HOUR
                        THEN 1 ELSE 0 END) AS imv,
               min(datediff('day', m.starttime, i.d)) FILTER (WHERE m.starttime <= i.d) AS mv_start_days
        FROM idx0 i LEFT JOIN mv m ON m.subject_id = i.subject_id
        GROUP BY 1, 2, 3, 4, 5
    """)
    con.execute("CREATE OR REPLACE TEMP TABLE cohort AS SELECT * FROM idx"
                + (" WHERE imv = 1" if imv_filter else ""))

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE idx_org_long AS
        SELECT DISTINCT c.cid, {_NORM.format(col='m.org_name')} AS org_norm,
               regexp_matches(upper(m.org_name), '{GN_OTHER_PATTERN}') AS is_other_gn,
               regexp_matches(upper(m.org_name), 'PSEUDOMONAS') AS is_pa
        FROM cohort c
        JOIN hosp.microbiologyevents m ON m.micro_specimen_id = c.cid
        WHERE m.org_name IS NOT NULL AND regexp_matches(m.org_name, '{GN_PATTERN}')
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE idx_org AS
        SELECT cid, max(is_pa::int) AS org_pa, max(is_other_gn::int) AS org_other,
               count(DISTINCT org_norm) AS n_org
        FROM idx_org_long GROUP BY 1
    """)

    scope_cond = _scope_sql(scope, "p.spec_type_desc")
    if scope != "all":
        scope_cond = scope_cond.replace("'RESPIRATORY_TRACT'", f"({RESP_SPECS})")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE prior AS
        SELECT c.cid AS icid,
               {_NORM.format(col='p.org_name')} AS org_norm,
               regexp_matches(upper(p.org_name), '{GN_PATTERN}') AS is_gn,
               p.ab_name AS ab, p.interpretation AS ph,
               (p.spec_type_desc IN ({RESP_SPECS})) AS is_resp
        FROM cohort c
        JOIN hosp.microbiologyevents p
          ON p.subject_id = c.subject_id
         AND p.org_name IS NOT NULL
         AND p.chartdate < c.d
         AND p.chartdate >= c.d - INTERVAL {int(history_days)} DAY
        WHERE {scope_cond}
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE prior_flag AS
        SELECT pr.*, (io.org_norm IS NOT NULL) AS is_same
        FROM prior pr
        LEFT JOIN idx_org_long io ON io.cid = pr.icid AND io.org_norm = pr.org_norm
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE f_hist AS
        SELECT c.cid,
               coalesce(max(CASE WHEN f.is_same THEN 1 ELSE 0 END), 0) AS prior_same_org,
               coalesce(max(CASE WHEN f.is_same AND f.ab = '{drug}'
                                  AND f.ph IN ('S', 'R') THEN 1 ELSE 0 END), 0) AS prior_same_ast,
               coalesce(max(CASE WHEN f.is_same AND f.ab = '{drug}'
                                  AND f.ph = 'R' THEN 1 ELSE 0 END), 0) AS prior_same_res,
               coalesce(max(CASE WHEN f.is_same AND f.ab = '{drug}'
                                  AND f.ph = 'R' AND f.is_resp THEN 1 ELSE 0 END), 0) AS prior_same_res_resp,
               coalesce(max(CASE WHEN f.is_same AND f.ab = '{drug}'
                                  AND f.ph = 'R' AND NOT f.is_resp THEN 1 ELSE 0 END), 0) AS prior_same_res_nonresp,
               coalesce(max(CASE WHEN f.is_gn AND NOT f.is_same THEN 1 ELSE 0 END), 0) AS prior_other_gn_org,
               coalesce(max(CASE WHEN f.is_gn AND NOT f.is_same AND f.ab = '{drug}'
                                  AND f.ph = 'R' THEN 1 ELSE 0 END), 0) AS prior_other_gn_res
        FROM cohort c LEFT JOIN prior_flag f ON f.icid = c.cid
        GROUP BY 1
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE cx_all AS
        SELECT DISTINCT subject_id, micro_specimen_id AS cid, chartdate AS cd
        FROM hosp.microbiologyevents
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE f_cx AS
        SELECT c.cid,
               count(DISTINCT p.cid) FILTER (WHERE p.cd < c.d AND p.cd >= c.d - INTERVAL 90 DAY)  AS n_cx_90d,
               count(DISTINCT p.cid) FILTER (WHERE p.cd < c.d AND p.cd >= c.d - INTERVAL 365 DAY) AS n_cx_365d,
               coalesce(datediff('day', max(p.cd) FILTER (WHERE p.cd < c.d), c.d), 9999) AS days_since_cx
        FROM cohort c LEFT JOIN cx_all p ON p.subject_id = c.subject_id
        GROUP BY 1, c.d
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE abx AS
        SELECT subject_id, lower(antibiotic) AS drug_name, starttime
        FROM derived.antibiotic WHERE starttime IS NOT NULL
    """)
    flags = ",\n".join(
        f"""coalesce(max(CASE WHEN regexp_matches(a.drug_name, '{pat}') THEN 1 ELSE 0 END)
                     FILTER (WHERE a.starttime < c.d
                             AND a.starttime >= c.d - INTERVAL 365 DAY), 0) AS {name}"""
        for name, pat in DRUGS.items()
    )
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE f_abx AS
        SELECT c.cid,
               coalesce(min(datediff('day', a.starttime, c.d)) FILTER (WHERE a.starttime < c.d), 9999) AS days_since_abx,
               coalesce(min(datediff('day', a.starttime, c.d)) FILTER (WHERE a.starttime < c.d
                     AND regexp_matches(a.drug_name, 'cefepime')), 9999) AS days_since_cefepime,
               count(DISTINCT a.drug_name) FILTER (WHERE a.starttime < c.d
                     AND a.starttime >= c.d - INTERVAL 365 DAY) AS n_drugs_365d,
               {flags}
        FROM cohort c LEFT JOIN abx a ON a.subject_id = c.subject_id
        GROUP BY 1
    """)

    df = con.execute("""
        SELECT c.subject_id AS patient_id, c.cid, c.hadm_id, c.y, c.d AS index_date,
               c.imv, c.mv_start_days,
               p.gender, p.anchor_age, p.anchor_year,
               CAST(substr(p.anchor_year_group, 1, 4) AS INTEGER) AS anchor_start,
               t.org_pa, t.org_other, t.n_org,
               x.n_cx_90d, x.n_cx_365d, x.days_since_cx,
               h.prior_same_org, h.prior_same_ast, h.prior_same_res,
               h.prior_same_res_resp, h.prior_same_res_nonresp,
               h.prior_other_gn_org, h.prior_other_gn_res,
               a.days_since_abx, a.days_since_cefepime, a.n_drugs_365d,
               a.d_fq, a.d_bsc, a.d_cefepime, a.d_carbapenem, a.d_piptazo,
               a.d_tmpsmx, a.d_nitro, a.d_amino, a.d_glyco
        FROM cohort c
        JOIN hosp.patients p USING (subject_id)
        LEFT JOIN f_hist  h ON h.cid = c.cid
        LEFT JOIN f_cx    x ON x.cid = c.cid
        LEFT JOIN f_abx   a ON a.cid = c.cid
        LEFT JOIN idx_org t ON t.cid = c.cid
        ORDER BY c.subject_id, c.d, c.cid
    """).df()

    df["age_exact"] = df["anchor_age"] + (df["index_date"].dt.year - df["anchor_year"])
    df["age_mid"] = df["age_exact"].map(band_age)
    df["male"] = (df["gender"] == "M").astype(int)
    df["year"] = df["anchor_start"] + (df["index_date"].dt.year - df["anchor_year"])
    df["system"] = "MIMIC-IV"
    df = df[(df["age_exact"] >= 18) & (df["year"].between(2008, 2022))]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """固定口径的缺失处理：sentinel 而非数据驱动插补。"""
    df = df.copy()
    for c in DEMO + ORG + ABX + CXBURDEN + HIST_SAME + HIST_OTHER + ["n_org", "days_since_cefepime"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if c in ("days_since_cx", "days_since_abx", "days_since_cefepime"):
            df[c] = df[c].fillna(SENTINEL)
        elif c == "age_mid":
            df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 60.0)
        elif c in ("n_cx_90d", "n_cx_365d", "n_drugs_365d"):
            df[c] = df[c].fillna(0.0)
        else:
            df[c] = df[c].fillna(0.0)
    return df

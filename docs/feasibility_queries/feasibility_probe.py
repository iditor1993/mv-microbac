
"""Feasibility probe for MV_microbac (read-only).

Reproduces the headline counts quoted in docs/feasibility_2026-10-24.md.

    python feasibility_probe.py

Reads both DuckDB databases read-only; prints a text report; writes nothing.
"""
from __future__ import annotations

import sys

import duckdb
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

pd.set_option("display.width", 200)

ARMD_DB = r"E:\clinical_research\ARMD_MGB_duckdb\ARMD_MGB.duckdb"
MIMIC_DB = r"E:\clinical_research\MIMIC-IV_3.1\mimiciv_v3_1.duckdb"

GN = ("PSEUDOMONAS|KLEBSIELLA|ESCHERICHIA|ENTEROBACTER|SERRATIA|PROTEUS"
      "|CITROBACTER|ACINETOBACTER|STENOTROPHOMONAS")
NORM = r"trim(regexp_replace(upper({c}), '\s*\(.*\)$', ''))"


def armd():
    con = duckdb.connect(ARMD_DB, read_only=True)
    con.execute("SET threads=8")
    q = lambda s: con.execute(s).df()

    # MV episodes reconstructed at patient level: culture date - days since procedure
    con.execute("""
        create or replace temp table mv_ep as
        select distinct anon_id,
               cast(order_time_jittered_utc_shifted as date) - to_days(procedure_days_culture) as mv_date
        from main.prior_procedures
        where procedure_description = 'mechvent' and procedure_days_culture is not null
    """)

    con.execute(f"""
        create or replace temp table micro as
        select anon_id, order_proc_id_coded as cid, organism,
               {NORM.format(c='organism')} as org_norm,
               antibiotic, CLSI_2022_pheno,
               order_time_jittered_utc_shifted as d
        from main.microbiology_cohort
        where culture_description = 'RESPIRATORY_TRACT' and organism is not null
          and regexp_matches(organism, '{GN}')
    """)

    con.execute(f"""
        create or replace temp table idx0 as
        select m.cid, m.anon_id, min(m.d) as d,
               max(case when m.CLSI_2022_pheno = 'Resistant' then 1 else 0 end) as y,
               max(case when m.organism = 'PSEUDOMONAS AERUGINOSA' then 1 else 0 end) as pa
        from micro m
        where m.antibiotic = 'cefepime'
          and m.CLSI_2022_pheno in ('Resistant', 'Susceptible')
          and regexp_matches(m.organism, '{GN}')
        group by 1, 2
    """)

    con.execute("""
        create or replace temp table idx as
        select i.*,
               min(datediff('day', e.mv_date, i.d)) filter (where e.mv_date <= i.d) as mv_days
        from idx0 i
        left join mv_ep e
          on e.anon_id = i.anon_id and e.mv_date <= i.d and e.mv_date >= i.d - interval 365 day
        group by 1, 2, 3, 4, 5
    """)

    print("== ARMD-MGB ==")
    print("GN + cefepime S/R respiratory cultures (no MV restriction):")
    print(q("select count(*) n, sum(y) events, round(avg(y), 3) rate from idx0").to_string(index=False))
    for w in (2, 7, 14, 30):
        print(q(f"""
            select {w} as mv_window_days, count(*) n, count(distinct anon_id) patients,
                   round(count(*) * 1.0 / count(distinct anon_id), 2) cultures_per_patient,
                   sum(y) events, round(avg(y), 3) event_rate
            from idx where mv_days <= {w}
        """).to_string(index=False))

    con.execute("create or replace temp table cohort as select * from idx where mv_days <= 7")

    con.execute(r"""
        create or replace temp table prior as
        select c.cid as icid, trim(regexp_replace(upper(p.organism), '\s*\(.*\)$', '')) as org_norm,
               p.antibiotic, p.CLSI_2022_pheno
        from cohort c
        join main.microbiology_cohort p
          on p.anon_id = c.anon_id and p.organism is not null
         and p.order_time_jittered_utc_shifted < c.d
         and p.order_time_jittered_utc_shifted >= c.d - interval 365 day
    """)

    con.execute("""
        create or replace temp table feat as
        select c.cid, c.y, c.pa,
               count(p.icid) > 0 as any_prior_cx,
               max(case when p.org_norm in (select org_norm from micro m where m.cid = c.cid)
                        then 1 else 0 end) as prior_same_org,
               max(case when p.org_norm in (select org_norm from micro m where m.cid = c.cid)
                         and p.antibiotic = 'cefepime'
                         and p.CLSI_2022_pheno in ('Resistant', 'Susceptible')
                        then 1 else 0 end) as prior_same_ast,
               max(case when p.org_norm in (select org_norm from micro m where m.cid = c.cid)
                         and p.antibiotic = 'cefepime'
                         and p.CLSI_2022_pheno = 'Resistant'
                        then 1 else 0 end) as prior_same_res
        from cohort c left join prior p on p.icid = c.cid
        group by 1, 2, 3
    """)

    print("\nprior-history prevalence, MV<=7d, 365-day window, any specimen:")
    print(q("""
        select count(*) n,
               round(avg(any_prior_cx::int), 3)      any_prior_cx,
               round(avg(prior_same_org::int), 3)    prior_same_org,
               round(avg(prior_same_ast::int), 3)    prior_same_ast,
               round(avg(prior_same_res::int), 3)    prior_same_res
        from feat
    """).to_string(index=False))

    print("\nprior same-organism cefepime-R x index culture:")
    print(q("""
        select prior_same_res, count(*) n, sum(y) events, round(avg(y), 3) rate
        from feat group by 1 order by 1
    """).to_string(index=False))

    print("\nany antibiotic exposure in prior 365 days:")
    print(q("""
        select round(avg((last_dose is not null and last_dose between 1 and 365)::int), 3) any_abx_365, count(*) n
        from (select c.cid,
                     min(a.last_dose_to_culture) filter (where a.last_dose_to_culture > 0) as last_dose
              from cohort c
              left join main.prior_abx a
                on a.anon_id = c.anon_id and a.order_proc_id_coded = c.cid
              group by 1) t
    """).to_string(index=False))

    print("\nindex organism mix (rows with cefepime AST, MV<=7d):")
    print(q(f"""
        select m.organism, count(*) n
        from main.microbiology_cohort m join cohort c on c.cid = m.order_proc_id_coded
        where m.antibiotic = 'cefepime' and m.CLSI_2022_pheno in ('Resistant', 'Susceptible')
        group by 1 order by 2 desc limit 8
    """).to_string(index=False))


def mimic():
    con = duckdb.connect(MIMIC_DB, read_only=True)
    con.execute("SET threads=8")
    q = lambda s: con.execute(s).df()

    con.execute(f"""
        create or replace temp table idx0 as
        select micro_specimen_id as cid, any_value(subject_id) as subject_id,
               any_value(hadm_id) as hadm_id, min(chartdate) as d,
               max(case when interpretation = 'R' then 1 else 0 end) as y,
               max(case when org_name = 'PSEUDOMONAS AERUGINOSA' then 1 else 0 end) as pa
        from hosp.microbiologyevents
        where spec_type_desc in ('SPUTUM','BRONCHOALVEOLAR LAVAGE','BRONCHIAL WASHINGS',
                                 'Mini-BAL','TRACHEAL ASPIRATE','BRONCHIAL BRUSH',
                                 'BRONCHIAL BRUSH - PROTECTED')
          and org_name is not null and regexp_matches(org_name, '{GN}')
          and ab_name = 'CEFEPIME' and interpretation in ('S', 'R')
        group by 1
    """)

    con.execute("""
        create or replace temp table mv as
        select s.subject_id, v.starttime, v.endtime
        from derived.ventilation v
        join (select subject_id, stay_id from icu.icustays) s using (stay_id)
        where v.ventilation_status in ('InvasiveVent', 'Tracheostomy')
    """)

    con.execute("""
        create or replace temp table idx as
        select i.*, max(case when m.starttime <= i.d + interval 12 hour
                              and m.endtime >= i.d - interval 12 hour then 1 else 0 end) as imv
        from idx0 i left join mv m on m.subject_id = i.subject_id
        group by 1, 2, 3, 4, 5, 6
    """)

    print("\n== MIMIC-IV ==")
    print("GN + cefepime S/R respiratory specimens:")
    print(q("select count(*) n, sum(y) events, round(avg(y), 3) rate from idx0").to_string(index=False))
    print("restricted to invasive ventilation +/-12h:")
    print(q("select count(*) n, count(distinct subject_id) patients, sum(y) events, round(avg(y), 3) rate from idx where imv = 1").to_string(index=False))

    con.execute("create or replace temp table cohort as select * from idx where imv = 1")
    con.execute(r"""
        create or replace temp table porg as
        select micro_specimen_id as cid,
               trim(regexp_replace(upper(org_name), '\s*\(.*\)$', '')) as org_norm
        from hosp.microbiologyevents
        where org_name is not null and regexp_matches(org_name,
              'PSEUDOMONAS|KLEBSIELLA|ESCHERICHIA|ENTEROBACTER|SERRATIA|PROTEUS'
              '|CITROBACTER|ACINETOBACTER|STENOTROPHOMONAS')
    """)
    con.execute(r"""
        create or replace temp table prior as
        select c.cid as icid,
               trim(regexp_replace(upper(p.org_name), '\s*\(.*\)$', '')) as org_norm,
               p.ab_name, p.interpretation
        from cohort c
        join hosp.microbiologyevents p
          on p.subject_id = c.subject_id and p.org_name is not null
         and p.chartdate < c.d and p.chartdate >= c.d - interval 365 day
    """)
    con.execute("""
        create or replace temp table feat as
        select c.cid, c.y,
               count(p.icid) > 0 as any_prior_cx,
               max(case when p.org_norm in (select org_norm from porg where cid = c.cid)
                        then 1 else 0 end) as prior_same_org,
               max(case when p.org_norm in (select org_norm from porg where cid = c.cid)
                         and p.ab_name = 'CEFEPIME' and p.interpretation in ('S', 'R')
                        then 1 else 0 end) as prior_same_ast,
               max(case when p.org_norm in (select org_norm from porg where cid = c.cid)
                         and p.ab_name = 'CEFEPIME' and p.interpretation = 'R'
                        then 1 else 0 end) as prior_same_res
        from cohort c left join prior p on p.icid = c.cid
        group by 1, 2
    """)
    print("\nprior-history prevalence, IMV cohort, 365-day window, any specimen:")
    print(q("""
        select count(*) n,
               round(avg(any_prior_cx::int), 3)   any_prior_cx,
               round(avg(prior_same_org::int), 3) prior_same_org,
               round(avg(prior_same_ast::int), 3) prior_same_ast,
               round(avg(prior_same_res::int), 3) prior_same_res
        from feat
    """).to_string(index=False))
    print("\nprior same-organism cefepime-R x index specimen:")
    print(q("select prior_same_res, count(*) n, sum(y) events, round(avg(y), 3) rate from feat group by 1 order by 1").to_string(index=False))
    print("\nany antibiotic exposure in prior 365 days:")
    print(q("""
        select round(avg((last_dose is not null and last_dose between 1 and 365)::int), 3) any_abx_365, count(*) n
        from (select c.cid,
                     min(datediff('day', a.starttime, c.d)) as last_dose
              from cohort c
              left join derived.antibiotic a
                on a.subject_id = c.subject_id and a.starttime < c.d
               and a.starttime >= c.d - interval 365 day
              group by 1) t
    """).to_string(index=False))

    print("\ntemporal split (anchor-based calendar year):")
    print(q("""
        with y as (
          select c.*, p.anchor_age, p.anchor_year,
                 cast(substr(p.anchor_year_group, 1, 4) as int) as anchor_start,
                 cast(strftime(c.d, '%Y') as int) as shifted_year
          from cohort c join hosp.patients p using (subject_id))
        select case when anchor_start + (shifted_year - anchor_year) <= 2014
                    then 'train 2008-2014' else 'test 2015-2022' end as split,
               count(*) n, sum(y) events, round(avg(y), 3) rate
        from y where anchor_start + (shifted_year - anchor_year) between 2008 and 2022
        group by 1 order by 1
    """).to_string(index=False))

    print("\nalternative phenotype - pneumonia ICD + ICU (severe pneumonia):")
    con.execute("""
        create or replace temp table pneu as
        select distinct hadm_id from hosp.diagnoses_icd
        where (icd_version = 10 and icd_code like 'J1%')
           or (icd_version = 9 and (icd_code like '48%' or icd_code like '507%'))
    """)
    print(q("""
        select count(*) n, sum(y) events, round(avg(y), 3) rate
        from idx0
        where hadm_id in (select hadm_id from pneu)
          and hadm_id in (select distinct hadm_id from icu.icustays)
    """).to_string(index=False))


if __name__ == "__main__":
    armd()
    mimic()

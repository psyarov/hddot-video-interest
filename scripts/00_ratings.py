"""Step 0 — interest ratings: PsychoPy CSVs -> one labelled table for the whole study.

Output (<derivatives>/group/ratings/):
    ratings_all.csv      one row per rated clip: session_tag, subject, session, block,
                         position_in_block, video_global, video_uid, rating, rating_z_session,
                         rating_z_subject, rating_pct_session, interest_tertile, rating_normative
    ratings_qc.csv       per session: presented / rated / expected clips
    agreement.csv        per subject: Spearman rho(own rating, leave-one-out normative rating)

    python scripts/00_ratings.py --config config/config.yaml
"""
import argparse

import pandas as pd

from dotinterest.config import group_dir, load_config, raw_path, sessions
from dotinterest.ratings import add_normalisations, agreement_summary, normative_rating, parse_psychopy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    out = group_dir(cfg, "ratings")

    frames, qcs = [], []
    for sub, ses in sessions(cfg):
        csv = raw_path(cfg, "ratings_csv", sub, ses)
        if not csv.exists():
            print(f"  {sub}/{ses}: no ratings file ({csv.name})")
            continue
        try:
            df, qc = parse_psychopy(csv, sub, ses, cfg["ratings"])
        except Exception as e:
            print(f"  {sub}/{ses}: FAILED {type(e).__name__}: {e}")
            continue
        frames.append(df)
        qcs.append(qc)
        flag = "" if qc["n_rated"] == qc["n_expected"] and not qc["duplicate_slots"] else "  <-- check"
        print(f"  {sub}/{ses}: {qc['n_rated']}/{qc['n_expected']} rated{flag}")
    if not frames:
        raise SystemExit("no rating files found")

    df = add_normalisations(pd.concat(frames, ignore_index=True))
    df["rating_normative"] = normative_rating(df)
    df.to_csv(out / "ratings_all.csv", index=False)
    pd.DataFrame(qcs).to_csv(out / "ratings_qc.csv", index=False)
    ag = agreement_summary(df)
    ag.to_csv(out / "agreement.csv", index=False)
    print(f"\n{len(df)} rated clips, {df.subject.nunique()} subjects, {df.session_tag.nunique()} sessions")
    if len(ag):
        print(f"agreement with others (median Spearman rho): {ag.rho.median():+.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

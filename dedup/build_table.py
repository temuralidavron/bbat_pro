import argparse
import csv
import os

from common import WORK, load_parts
from match import range_pairs, UF

HEADER = ["group_no", "in_group_no", "original_id", "file_url", "sim"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--out-dir", default=WORK)
    args = ap.parse_args()

    emb, ids, urls = load_parts()
    print(f"embeddings={emb.shape[0]}")

    pairs = range_pairs(emb, args.threshold)
    uf = UF(emb.shape[0])
    member_sim = [0.0] * emb.shape[0]
    for (a, b), s in pairs.items():
        uf.union(a, b)
        member_sim[a] = max(member_sim[a], s)
        member_sim[b] = max(member_sim[b], s)

    comps = {}
    for x in range(emb.shape[0]):
        comps.setdefault(uf.find(x), []).append(x)
    groups = [m for m in comps.values() if len(m) >= 2]
    groups.sort(key=lambda m: (-len(m), min(int(ids[x]) for x in m)))

    rows = []
    summary = []
    for gno, members in enumerate(groups, 1):
        members = sorted(members, key=lambda x: int(ids[x]))
        sims = [member_sim[x] for x in members]
        for ino, x in enumerate(members, 1):
            rows.append([gno, ino, ids[x], urls[x], round(member_sim[x], 4)])
        summary.append([gno, len(members), round(max(sims), 4), round(min(sims), 4)])

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "duplicates.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)

    import pandas as pd
    xlsx_path = os.path.join(args.out_dir, "duplicates.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        pd.DataFrame(rows, columns=HEADER).to_excel(xw, sheet_name="duplicates", index=False)
        pd.DataFrame(summary, columns=["group_no", "size", "max_sim", "min_sim"]).to_excel(
            xw, sheet_name="summary", index=False)

    mx = max((len(m) for m in groups), default=0)
    print(f"groups={len(groups)} duplicate_rows={len(rows)} max_group={mx}")
    print(f"-> {csv_path}")
    print(f"-> {xlsx_path} (duplicates + summary)")


if __name__ == "__main__":
    main()

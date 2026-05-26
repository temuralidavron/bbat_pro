import argparse
import csv
import html
import os
import re

import numpy as np

from common import WORK, load_parts

_DATE_RE = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/")


def url_date(u):
    m = _DATE_RE.search(u or "")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


def groups_at(emb, A, B, SIM, threshold):
    m = SIM >= threshold
    a = A[m].tolist()
    b = B[m].tolist()
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(a)):
        ra, rb = find(a[i]), find(b[i])
        if ra != rb:
            parent[ra] = rb

    comp = {}
    for x in parent:
        comp.setdefault(find(x), []).append(x)

    groups = []
    for members in comp.values():
        if len(members) < 2:
            continue
        idx = np.array(members)
        c = emb[idx].mean(0)
        c /= max(np.linalg.norm(c), 1e-9)
        keep = idx[(emb[idx] @ c) >= threshold]
        if len(keep) < 2:
            continue
        ks = (emb[keep] @ c).astype(float)
        groups.append((keep.tolist(), ks.tolist()))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--pairs", default=os.path.join(WORK, "pairs.npz"))
    ap.add_argument("--out-dir", default=WORK)
    ap.add_argument("--html-limit", type=int, default=800, help="max groups rendered in HTML (CSV/XLSX always full)")
    args = ap.parse_args()

    emb, ids, urls = load_parts()
    print(f"embeddings={emb.shape[0]}", flush=True)
    d = np.load(args.pairs)
    A, B, SIM = d["a"], d["b"], d["sim"]

    raw = groups_at(emb, A, B, SIM, args.threshold)
    groups = []
    for members, sims in raw:
        rows = sorted(
            ({"id": ids[k], "url": urls[k], "date": url_date(urls[k]), "sim": round(float(s), 4)}
             for k, s in zip(members, sims)),
            key=lambda r: -r["sim"])
        groups.append(rows)
    groups.sort(key=lambda rows: (-len(rows), -max(r["sim"] for r in rows)))

    thr = args.threshold
    base = os.path.join(args.out_dir, f"duplicates_{thr}")
    header = ["group_no", "in_group_no", "id", "file_url", "date", "sim"]

    with open(base + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for gno, rows in enumerate(groups, 1):
            for ino, r in enumerate(rows, 1):
                w.writerow([gno, ino, r["id"], r["url"], r["date"], r["sim"]])

    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "duplicates"
    ws.append(header)
    for gno, rows in enumerate(groups, 1):
        for ino, r in enumerate(rows, 1):
            ws.append([gno, ino, r["id"], r["url"], r["date"], r["sim"]])
    ws2 = wb.create_sheet("summary")
    ws2.append(["group_no", "size", "max_sim", "min_sim"])
    for gno, rows in enumerate(groups, 1):
        s = [r["sim"] for r in rows]
        ws2.append([gno, len(rows), max(s), min(s)])
    wb.save(base + ".xlsx")

    rows_total = sum(len(r) for r in groups)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>dublikatlar</title><style>",
        "body{font-family:sans-serif;background:#0f1115;color:#e6e6e6;margin:0;padding:18px}",
        "h2{font-size:18px}.g{background:#1a1d24;border-radius:10px;padding:12px;margin-bottom:10px}",
        ".g b{font-size:14px}.m{display:inline-block;text-align:center;margin:4px;vertical-align:top}",
        ".m img{width:110px;height:110px;object-fit:cover;border:1px solid #333;border-radius:6px}",
        ".c{font-size:11px;color:#9aa4b2}</style></head><body>",
        f"<h2>Dublikatlar (threshold &ge; {thr}) &mdash; jami {len(groups)} guruh, "
        f"{rows_total} ta ID. Quyida birinchi {min(args.html_limit, len(groups))} guruh "
        f"(to'liq ro'yxat CSV/XLSX'da).</h2>",
    ]
    for gno, rows in enumerate(groups[:args.html_limit], 1):
        parts.append(f"<div class='g'><b>Guruh #{gno}</b> &nbsp;({len(rows)} ta ID)<br>")
        for r in rows:
            parts.append(f"<div class='m'><img loading='lazy' src='{html.escape(r['url'])}'>"
                         f"<div class='c'>{r['id']}<br>{r['date']}<br>sim {r['sim']}</div></div>")
        parts.append("</div>")
    parts.append("</body></html>")
    with open(base + ".html", "w") as f:
        f.write("".join(parts))

    print(f"groups={len(groups)} duplicate_ids={rows_total}", flush=True)
    print(f"-> {base}.csv", flush=True)
    print(f"-> {base}.xlsx", flush=True)
    print(f"-> {base}.html", flush=True)


if __name__ == "__main__":
    main()

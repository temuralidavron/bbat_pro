import io
import os
import re
import csv
import html
import sys
from functools import lru_cache

import numpy as np
import django
from django.conf import settings
from django.http import HttpResponse
from django.urls import path

settings.configure(
    DEBUG=True,
    SECRET_KEY="dev-bbat",
    ROOT_URLCONF=__name__,
    ALLOWED_HOSTS=["*"],
    INSTALLED_APPS=[],
    DATABASES={},
    MIDDLEWARE=[],
    TEMPLATES=[],
)
django.setup()

from common import WORK, load_parts

_DATA = {}
_DATE_RE = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/")


def data():
    if not _DATA:
        emb, ids, urls = load_parts()
        npz = np.load(os.path.join(WORK, "pairs.npz"))
        _DATA.update(emb=emb, ids=ids, urls=urls,
                     A=npz["a"], B=npz["b"], SIM=npz["sim"])
        print(f"[webapp] loaded {emb.shape[0]} embeddings, {len(_DATA['SIM'])} pairs", flush=True)
    return _DATA


def url_date(u):
    m = _DATE_RE.search(u or "")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


@lru_cache(maxsize=16)
def groups_at(threshold):
    """Union-Find at threshold, then centroid-cohesion to drop chain members."""
    d = data()
    A, B, SIM, emb = d["A"], d["B"], d["SIM"], d["emb"]
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
        sims = emb[idx] @ c
        keep = idx[sims >= threshold]
        if len(keep) < 2:
            continue
        ks = (emb[keep] @ c).astype(float)
        groups.append((keep.tolist(), ks.tolist()))
    return groups


def build_groups(threshold):
    d = data()
    ids, urls = d["ids"], d["urls"]
    out = []
    for members, sims in groups_at(threshold):
        rows = []
        for k, s in zip(members, sims):
            rows.append({"id": ids[k], "url": urls[k], "date": url_date(urls[k]), "sim": round(float(s), 4)})
        rows.sort(key=lambda r: -r["sim"])
        gsims = [r["sim"] for r in rows]
        out.append({
            "members": rows,
            "size": len(rows),
            "max_sim": max(gsims),
            "min_sim": min(gsims),
            "max_date": max((r["date"] for r in rows), default=""),
            "min_id": min(int(r["id"]) for r in rows),
        })
    return out


SORT_KEYS = {
    "size": lambda g: g["size"],
    "threshold": lambda g: g["max_sim"],
    "date": lambda g: g["max_date"],
    "id": lambda g: g["min_id"],
}


def filtered(threshold, min_size, q, sort, direction):
    groups = build_groups(threshold)
    total = len(groups)
    sel = groups
    if min_size > 2:
        sel = [g for g in sel if g["size"] >= min_size]
    if q:
        sel = [g for g in sel if any(q in m["id"] for m in g["members"])]
    keyfn = SORT_KEYS.get(sort, SORT_KEYS["size"])
    sel = sorted(sel, key=keyfn, reverse=(direction == "desc"))
    return total, sel


def _params(request):
    g = request.GET
    try:
        threshold = round(float(g.get("threshold", 0.65)), 2)
    except ValueError:
        threshold = 0.65
    threshold = min(0.99, max(0.30, threshold))
    try:
        min_size = int(g.get("min_size", 2))
    except ValueError:
        min_size = 2
    return {
        "threshold": threshold,
        "min_size": max(2, min_size),
        "q": g.get("q", "").strip(),
        "sort": g.get("sort", "size"),
        "dir": g.get("dir", "desc"),
    }


def index(request):
    p = _params(request)
    total, sel = filtered(p["threshold"], p["min_size"], p["q"], p["sort"], p["dir"])
    limit = 60
    shown = sel[:limit]

    qs = (f"threshold={p['threshold']}&min_size={p['min_size']}&q={html.escape(p['q'])}"
          f"&sort={p['sort']}&dir={p['dir']}")

    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>bbat dublikatlar</title>",
             "<style>",
             "body{font-family:sans-serif;background:#0f1115;color:#e6e6e6;margin:0;padding:20px}",
             "h1{font-size:20px}.bar{background:#1a1d24;padding:16px;border-radius:10px;margin-bottom:16px}",
             "label{font-size:13px;color:#9aa4b2;margin-right:6px}",
             "input,select{background:#0f1115;color:#e6e6e6;border:1px solid #333;border-radius:6px;padding:6px 8px}",
             ".btn{background:#2d6cdf;color:#fff;border:none;border-radius:6px;padding:8px 14px;cursor:pointer;text-decoration:none;display:inline-block}",
             ".btn.alt{background:#3a3f4b}",
             ".count{font-size:14px;color:#9aa4b2;margin:10px 0}",
             ".g{background:#1a1d24;border-radius:10px;padding:12px;margin-bottom:12px}",
             ".g h3{margin:0 0 8px;font-size:14px;font-weight:600}",
             ".m{display:inline-block;text-align:center;margin:4px;vertical-align:top}",
             ".m img{width:110px;height:110px;object-fit:cover;border:1px solid #333;border-radius:6px}",
             ".m .c{font-size:11px;color:#9aa4b2}",
             ".pill{background:#22262f;border-radius:20px;padding:2px 10px;font-size:12px;color:#9aa4b2;margin-left:6px}",
             "</style></head><body>",
             "<h1>bbat — dublikat odamlar</h1>",
             "<form class='bar' method='get'>",
             f"<label>Threshold</label>"
             f"<input type='range' min='0.30' max='0.99' step='0.01' value='{p['threshold']}' "
             "oninput=\"document.getElementById('thr').value=this.value\"> ",
             f"<input type='number' id='thr' name='threshold' min='0.30' max='0.99' step='0.01' "
             f"value='{p['threshold']}' style='width:80px'> ",
             f"<label>Min hajm</label><input type='number' name='min_size' min='2' value='{p['min_size']}' style='width:70px'> ",
             f"<label>ID qidirish</label><input type='text' name='q' value='{html.escape(p['q'])}' placeholder='id...' style='width:120px'> ",
             "<label>Saralash</label><select name='sort'>"]
    for key, lbl in [("size", "Hajm"), ("threshold", "O'xshashlik"), ("date", "Sana"), ("id", "ID")]:
        sel_attr = " selected" if p["sort"] == key else ""
        parts.append(f"<option value='{key}'{sel_attr}>{lbl}</option>")
    parts.append("</select> <select name='dir'>")
    for key, lbl in [("desc", "kamayish"), ("asc", "o'sish")]:
        sel_attr = " selected" if p["dir"] == key else ""
        parts.append(f"<option value='{key}'{sel_attr}>{lbl}</option>")
    parts.append("</select> ")
    parts.append("<button class='btn' type='submit'>Qo'llash</button> ")
    parts.append(f"<a class='btn alt' href='/export?{qs}&format=csv'>CSV</a> ")
    parts.append(f"<a class='btn alt' href='/export?{qs}&format=xlsx'>XLSX</a>")
    parts.append("</form>")

    parts.append(f"<div class='count'>Natija: <b>{len(sel)}</b> / Jami: <b>{total}</b> guruh "
                 f"(threshold {p['threshold']}). Quyida birinchi {min(limit, len(sel))} tasi.</div>")

    for gi, g in enumerate(shown, 1):
        parts.append("<div class='g'><h3>"
                     f"Guruh #{gi} <span class='pill'>{g['size']} ta ID</span>"
                     f"<span class='pill'>min sim {g['min_sim']}</span>"
                     f"<span class='pill'>max sim {g['max_sim']}</span>"
                     f"<span class='pill'>{g['max_date']}</span></h3>")
        for m in g["members"]:
            parts.append(f"<div class='m'><img loading='lazy' src='{html.escape(m['url'])}'>"
                         f"<div class='c'>{m['id']}<br>{m['date']}<br>sim {m['sim']}</div></div>")
        parts.append("</div>")

    parts.append("</body></html>")
    return HttpResponse("".join(parts))


def export(request):
    p = _params(request)
    fmt = request.GET.get("format", "csv")
    _, sel = filtered(p["threshold"], p["min_size"], p["q"], p["sort"], p["dir"])

    header = ["group_no", "in_group_no", "id", "file_url", "date", "sim"]
    rows = []
    for gno, g in enumerate(sel, 1):
        for ino, m in enumerate(g["members"], 1):
            rows.append([gno, ino, m["id"], m["url"], m["date"], m["sim"]])

    name = f"duplicates_thr{p['threshold']}"
    if fmt == "xlsx":
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "duplicates"
        ws.append(header)
        for r in rows:
            ws.append(r)
        ws2 = wb.create_sheet("summary")
        ws2.append(["group_no", "size", "max_sim", "min_sim", "max_date"])
        for gno, g in enumerate(sel, 1):
            ws2.append([gno, g["size"], g["max_sim"], g["min_sim"], g["max_date"]])
        buf = io.BytesIO()
        wb.save(buf)
        resp = HttpResponse(buf.getvalue(),
                            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        resp["Content-Disposition"] = f'attachment; filename="{name}.xlsx"'
        return resp

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    resp = HttpResponse(buf.getvalue(), content_type="text/csv")
    resp["Content-Disposition"] = f'attachment; filename="{name}.csv"'
    return resp


urlpatterns = [
    path("", index),
    path("export", export),
]

if __name__ == "__main__":
    from django.core.management import execute_from_command_line
    data()
    execute_from_command_line(sys.argv)

"""In-memory request log + live HTML view for Laptop 1 (Issue #30).

Every call from Laptop 2 is kept in a bounded ring buffer and shown at
``GET /logs/view`` (auto-refreshing table) and ``GET /logs`` (JSON), in
addition to the terminal / api/logs/requests.log lines.
"""
from __future__ import annotations

import threading
from collections import deque
from datetime import datetime

MAX_ENTRIES = 500
_entries: deque[dict] = deque(maxlen=MAX_ENTRIES)
_lock = threading.Lock()
_counter = 0


def record(**fields) -> None:
    global _counter
    with _lock:
        _counter += 1
        _entries.appendleft({"id": _counter, "time": datetime.now().strftime("%H:%M:%S"), **fields})


def recent(limit: int = 200) -> dict:
    with _lock:
        rows = list(_entries)[: max(1, min(limit, MAX_ENTRIES))]
        clients = sorted({e.get("client", "?") for e in _entries})
        return {"total": _counter, "clients": clients, "entries": rows}


def clear() -> None:
    global _counter
    with _lock:
        _entries.clear()
        _counter = 0


LOG_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PathSnap API — live request log</title>
<style>
  :root { --bg:#0f172a; --line:#1e2b47; --text:#e2e8f0; --muted:#94a3b8;
          --ok:#22c55e; --warn:#f59e0b; --err:#ef4444; --accent:#38bdf8; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui, sans-serif; }
  header { padding:16px 20px; border-bottom:1px solid var(--line); display:flex; gap:24px;
           align-items:center; flex-wrap:wrap; }
  h1 { font-size:18px; margin:0; }
  .dot { width:10px; height:10px; border-radius:50%; background:var(--ok); display:inline-block;
         margin-right:6px; animation:p 1.5s infinite; }
  @keyframes p { 50% { opacity:.3 } }
  .stat { color:var(--muted); } .stat b { color:var(--text); font-size:16px; }
  main { padding:12px 20px; overflow-x:auto; } table { width:100%; border-collapse:collapse; }
  th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
  th { color:var(--muted); font-weight:600; position:sticky; top:0; background:var(--bg); }
  tr.new { animation:flash 1.2s; } @keyframes flash { from { background:#1e3a5f } to { background:transparent } }
  .s2 { color:var(--ok); } .s4 { color:var(--warn); } .s5 { color:var(--err); }
  .m { font-family:ui-monospace, Consolas, monospace; } .muted { color:var(--muted); }
  .pill { padding:2px 8px; border-radius:999px; background:#1e293b; }
  .predict { color:var(--accent); font-weight:600; }
</style></head><body>
<header>
  <h1><span class="dot"></span>PathSnap API — live request log (Laptop 1)</h1>
  <span class="stat">calls <b id="total">0</b></span>
  <span class="stat">clients <b id="clients">–</b></span>
  <span class="stat">last call <b id="last">–</b></span>
  <span class="stat muted">auto-refresh every 1.5 s · JSON at /logs</span>
</header>
<main><table>
  <thead><tr><th>#</th><th>time</th><th>client (Laptop 2)</th><th>request</th><th>status</th>
  <th>latency</th><th>trip</th><th>model</th><th>classifier</th><th>result</th></tr></thead>
  <tbody id="rows"><tr><td colspan="10" class="muted">waiting for requests…</td></tr></tbody>
</table></main>
<script>
let lastId = 0;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
async function refresh() {
  try {
    const r = await fetch("/logs?limit=200", {cache: "no-store"});
    const d = await r.json();
    document.getElementById("total").textContent = d.total;
    document.getElementById("clients").textContent = d.clients.join(", ") || "–";
    document.getElementById("last").textContent = d.entries.length ? d.entries[0].time : "–";
    if (!d.entries.length) return;
    document.getElementById("rows").innerHTML = d.entries.map(e => {
      const cls = "s" + String(e.status)[0];
      const isNew = e.id > lastId ? ' class="new"' : "";
      const req = e.path === "/predict" ? `<span class="predict">${esc(e.method)} ${esc(e.path)}</span>`
                                        : `${esc(e.method)} ${esc(e.path)}`;
      return `<tr${isNew}><td class="m muted">${e.id}</td><td class="m">${esc(e.time)}</td>
        <td class="m">${esc(e.client)}</td><td class="m">${req}</td><td class="m ${cls}">${esc(e.status)}</td>
        <td class="m">${esc(e.latency_ms)} ms</td><td class="m">${esc(e.trip_id || "")}</td>
        <td>${e.model ? `<span class="pill">${esc(e.model)}</span>` : ""}</td>
        <td class="m">${esc(e.classifier || "")}</td>
        <td class="muted">${esc(e.summary || "")}</td></tr>`;
    }).join("");
    lastId = d.entries[0].id;
  } catch (err) { document.getElementById("last").textContent = "API not reachable"; }
}
refresh(); setInterval(refresh, 1500);
</script></body></html>
"""

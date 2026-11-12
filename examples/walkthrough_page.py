"""
Shared page shell for the workflow walkthroughs (run_*_walkthrough.py).

A walkthrough template holds only its own markup and rendering code. It marks
three places for this module to fill: ``/*BASE_CSS*/`` (palette, layout, tables,
pills, bars), ``/*BASE_JS*/`` (formatting helpers and the mini-ATM diagram) and
``/*DATA*/`` (the JSON the script collected).
"""

import json
from pathlib import Path

PAGE_HEAD = ('<!doctype html>\n<meta charset="utf-8">\n'
             '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
             '<style>body{margin:0}[hidden]{display:none!important}</style>\n')

BASE_CSS = r"""
:root {
  --bg: #f2f5f7; --surface: #ffffff; --ink: #16212b; --muted: #5a6877; --line: #cbd4dc; --line-soft: #e3e8ed;
  --accent: #0b6a94; --accent-soft: #d9ebf3;
  --fail: #bf3b2b; --fail-soft: #f9ddd7;
  --ok: #2b7a4b; --ok-soft: #dcefe3;
  --warn: #9a6a12; --warn-soft: #f7e8c6;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", sans-serif;
  --body: "IBM Plex Sans", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f151b; --surface: #17202a; --ink: #e5ecf2; --muted: #94a2b1; --line: #344352; --line-soft: #243140;
  --accent: #5fbbe0; --accent-soft: #15364a;
  --fail: #f07360; --fail-soft: #4b221c;
  --ok: #63c08a; --ok-soft: #173826;
  --warn: #e3b550; --warn-soft: #43350f; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #0f151b; --surface: #17202a; --ink: #e5ecf2; --muted: #94a2b1; --line: #344352; --line-soft: #243140;
  --accent: #5fbbe0; --accent-soft: #15364a;
  --fail: #f07360; --fail-soft: #4b221c;
  --ok: #63c08a; --ok-soft: #173826;
  --warn: #e3b550; --warn-soft: #43350f; color-scheme: dark }

body { background: var(--bg); color: var(--ink); font-family: var(--body); font-size: 15px; line-height: 1.55; }
body::before { content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10; background: var(--bg); }
.wrap { max-width: 1180px; margin: 0 auto; padding-block: 28px 56px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: flex; flex-direction: column; gap: 46px; }
h1, h2 { font-family: var(--display); text-wrap: balance; margin: 0; line-height: 1.15; }
h1 { font-size: 2.3rem; font-weight: 700; letter-spacing: -0.01em; }
h2 { font-size: 1.55rem; font-weight: 600; }
p { margin: 0; max-width: 70ch; }
code, .mono { font-family: var(--mono); font-size: 0.88em; }
.lede { font-size: 1.05rem; color: var(--muted); margin-top: 10px; }
.eyebrow, .cap { font-family: var(--mono); font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; }
.eyebrow { color: var(--accent); margin-bottom: 6px; } .cap { color: var(--muted); font-weight: 600; }
section { display: flex; flex-direction: column; gap: 16px; }
.sec-head { display: flex; flex-direction: column; gap: 8px; }
.sec-head p, .note { color: var(--muted); } .note { font-size: 0.9rem; }
.sec-head p b, .note b, li b { color: var(--ink); font-weight: 600; }
.box { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; min-width: 0; }
.pad { padding: 14px 16px; display: flex; flex-direction: column; gap: 10px; }
.scroll { overflow-x: auto; }
.two { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr); gap: 20px; align-items: start; }
.half { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; align-items: start; }
.three { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; align-items: start; }
@media (max-width: 900px) { .two, .half, .three { grid-template-columns: minmax(0, 1fr); } }
ul.plain { margin: 0; padding-left: 1.15em; display: flex; flex-direction: column; gap: 8px; max-width: 78ch; color: var(--muted); }
.fig svg { display: block; width: 100%; min-width: 600px; height: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.86rem; }
th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--line-soft); white-space: nowrap; }
th { font: 600 0.7rem var(--mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
tr:last-child td { border-bottom: 0; }
td.num, th.num, td.h { font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; }
td.c, th.c { text-align: center; }
td.wrapcell { white-space: normal; min-width: 15em; }
tbody tr.pick { cursor: pointer; } tbody tr.pick:hover td { background: var(--line-soft); } tbody tr.on td { background: var(--accent-soft); }
.pill { display: inline-block; font: 600 0.68rem var(--mono); letter-spacing: 0.05em; text-transform: uppercase; padding: 2px 7px; border-radius: 3px; border: 1px solid var(--line); color: var(--muted); white-space: nowrap; }
.pill.critical { background: var(--fail); border-color: var(--fail); color: var(--surface); }
.pill.high, .pill.no { background: var(--fail-soft); border-color: var(--fail); color: var(--fail); }
.pill.medium, .pill.warn { background: var(--warn-soft); border-color: var(--warn); color: var(--warn); }
.pill.ok { background: var(--ok-soft); border-color: var(--ok); color: var(--ok); }
.pill.accent { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
.bar { display: grid; grid-template-columns: 3.4em minmax(40px, 1fr); gap: 8px; align-items: center; font-family: var(--mono); font-variant-numeric: tabular-nums; }
.track { height: 9px; background: var(--line-soft); border-radius: 2px; overflow: hidden; }
.fill { height: 100%; background: var(--accent); display: block; } .fill.f { background: var(--fail); }
.card { padding: 12px 14px; display: flex; flex-direction: column; gap: 6px; }
.card .rk { font: 600 0.74rem var(--mono); color: var(--accent); letter-spacing: 0.05em; }
.card .who { font: 600 1.05rem var(--display); }
.card .join { border-top: 1px dashed var(--line); padding-top: 8px; margin-top: 2px; display: flex; flex-direction: column; gap: 5px; font-size: 0.86rem; color: var(--muted); }
.score { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.score .val { font: 600 1.9rem var(--mono); line-height: 1; } .score .sub { font-size: 0.82rem; color: var(--muted); }
.keys { display: flex; flex-wrap: wrap; gap: 6px 16px; padding: 8px 12px; border-bottom: 1px solid var(--line-soft); font-size: 0.8rem; color: var(--muted); align-items: center; }
.key { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.ln { width: 22px; height: 0; border-top: 1.6px solid var(--muted); display: inline-block; } .ln.dash { border-top-style: dashed; } .ln.dot { border-top-style: dotted; }
.sw { width: 12px; height: 12px; display: inline-block; border-radius: 2px; background: var(--accent); }
.picker { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--bg); border-bottom: 1px solid var(--line); padding-block: 10px; display: flex; gap: 14px; align-items: center; }
.chips { display: flex; gap: 6px; overflow-x: auto; min-width: 0; padding-bottom: 2px; }
.chip { font: 500 0.82rem var(--body); color: var(--ink); background: var(--surface); border: 1px solid var(--line); border-radius: 4px; padding: 5px 10px; cursor: pointer; white-space: nowrap; }
.chip:hover { border-color: var(--accent); }
.chip[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: var(--surface); }
button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.formula { font-family: var(--mono); font-size: 0.86rem; background: var(--bg); border: 1px solid var(--line-soft); border-radius: 4px; padding: 8px 10px; overflow-x: auto; white-space: nowrap; }
footer { border-top: 1px solid var(--line); padding-top: 16px; font-size: 0.84rem; color: var(--muted); display: flex; flex-direction: column; gap: 8px; }
footer p { max-width: 92ch; }

svg text { font-family: var(--body); fill: var(--ink); }
.n rect, .n polygon { fill: var(--surface); stroke: var(--muted); stroke-width: 1.3; }
.n.host > rect { fill: var(--bg); stroke: var(--line); stroke-dasharray: 5 3; }
.n.lib rect { stroke-dasharray: 3 2; }
.n.pickable { cursor: pointer; }
.n.sel > rect, .n.sel > polygon { stroke: var(--accent); stroke-width: 2.6; }
.n.hot > rect, .n.hot > polygon { fill: var(--fail-soft); stroke: var(--fail); stroke-width: 1.8; }
.n.mark > rect, .n.mark > polygon { fill: var(--accent-soft); stroke: var(--accent); stroke-width: 1.8; }
.n.dim { opacity: 0.4; }
.nm { font-size: 12px; font-weight: 600; text-anchor: middle; }
.sb { font-family: var(--mono); font-size: 9.5px; fill: var(--muted); text-anchor: middle; }
.hl { font-family: var(--mono); font-size: 11px; fill: var(--muted); }
.bdg rect { fill: var(--ink); } .bdg text { fill: var(--bg); font-family: var(--mono); font-size: 9.5px; font-weight: 600; text-anchor: middle; }
.e { stroke: var(--muted); stroke-width: 1.2; fill: none; } .e.routes { stroke-dasharray: 5 4; } .e.uses { stroke-dasharray: 1.5 3.5; stroke-linecap: round; stroke-width: 1.6; }
.ah { fill: var(--muted); }
.link { stroke: var(--muted); stroke-width: 3; }
.lbl { font-family: var(--mono); font-size: 10px; fill: var(--muted); }
.axis { stroke: var(--line); } .grid { stroke: var(--line-soft); } .diag { stroke: var(--muted); stroke-dasharray: 4 3; }
.pt { fill: var(--accent); fill-opacity: 0.6; stroke: var(--accent); }
.ptl { font-family: var(--mono); font-size: 10px; paint-order: stroke; stroke: var(--surface); stroke-width: 3px; }
"""

BASE_JS = r"""
const NS = 'http://www.w3.org/2000/svg';
const NAME = { A1: 'radar', A2: 'tracker', A3: 'conflict-detector', A4: 'flight-plan', A5: 'display', A6: 'recorder', B1: 'sensor-broker', B2: 'ops-broker', T1: 'radar.plots', T2: 'tracks', T3: 'conflict.alerts', T4: 'flight.plans', N1: 'sensor-host', N2: 'ops-host', L1: 'track-model' };
const ORDER = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'L1', 'B1', 'B2', 'N1', 'N2', 'T1', 'T2', 'T3', 'T4'];
const f2 = v => v == null ? '–' : v.toFixed(2), f3 = v => v == null ? '–' : v.toFixed(3), f4 = v => v == null ? '–' : v.toFixed(4);
const esc = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const who = id => `<span class="mono">${esc(id)}</span> ${esc(NAME[id] || '')}`;
const sum = a => a.reduce((x, y) => x + y, 0);
function el(tag, attrs, parent, text) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; if (parent) parent.appendChild(e); return e; }
const heat = (v, max) => `background: color-mix(in srgb, var(--accent) ${Math.round(Math.max(0, Math.min(1, max ? v / max : 0)) * 55)}%, transparent)`;
const bar = (v, max, cls) => `<span class="bar"><span>${f3(v)}</span><span class="track"><span class="fill ${cls || ''}" style="width:${Math.max(0, Math.min(1, max ? v / max : 0)) * 100}%"></span></span></span>`;
const LEVEL = { critical: 'critical', high: 'high', medium: 'medium' };
const level = l => `<span class="pill ${LEVEL[String(l).toLowerCase()] || ''}">${esc(String(l).toLowerCase())}</span>`;
const verdict = (ok, yes, no) => ok == null ? '<span class="pill">not measured</span>' : `<span class="pill ${ok ? 'ok' : 'no'}">${ok ? (yes || 'pass') : (no || 'fail')}</span>`;

/* The mini-ATM raw structural graph. state: {id: 'sel' | 'hot' | 'mark' | 'dim'}, badges: {id: text}, sub: {id: text}. */
const MINI = {
  B1: { t: 'Broker', x: 210, y: 46, w: 110, h: 36 }, B2: { t: 'Broker', x: 490, y: 46, w: 110, h: 36 },
  T1: { t: 'Topic', x: 130, y: 150, w: 100, h: 36 }, T2: { t: 'Topic', x: 290, y: 150, w: 100, h: 36 }, T3: { t: 'Topic', x: 530, y: 150, w: 100, h: 36 }, T4: { t: 'Topic', x: 650, y: 150, w: 100, h: 36 },
  A1: { t: 'Application', x: 68, y: 282, w: 100, h: 38 }, A2: { t: 'Application', x: 193, y: 282, w: 100, h: 38 }, A6: { t: 'Application', x: 350, y: 282, w: 100, h: 38 },
  A3: { t: 'Application', x: 470, y: 282, w: 106, h: 38 }, A5: { t: 'Application', x: 590, y: 282, w: 100, h: 38 }, A4: { t: 'Application', x: 710, y: 282, w: 100, h: 38 },
  L1: { t: 'Library', x: 330, y: 388, w: 100, h: 36 },
};
const MINI_HOST = { N1: { x: 8, y: 248, w: 242, h: 92 }, N2: { x: 288, y: 248, w: 484, h: 92 } };
const MINI_EDGES = [['A1', 'T1', 'flow'], ['A2', 'T2', 'flow'], ['A3', 'T3', 'flow'], ['A4', 'T4', 'flow'], ['T1', 'A2', 'flow'], ['T2', 'A3', 'flow'], ['T2', 'A5', 'flow'], ['T3', 'A5', 'flow'], ['T4', 'A5', 'flow'], ['T1', 'A6', 'flow'], ['T2', 'A6', 'flow'], ['T4', 'A6', 'flow'],
  ['B1', 'T1', 'routes'], ['B1', 'T2', 'routes'], ['B2', 'T2', 'routes'], ['B2', 'T3', 'routes'], ['B2', 'T4', 'routes'], ['A2', 'L1', 'uses'], ['A3', 'L1', 'uses']];
function drawMiniAtm(svg, opts = {}) {
  const { state = {}, badges = {}, sub = {}, pickable = false } = opts;
  svg.textContent = ''; svg.setAttribute('viewBox', '0 0 780 420');
  const badge = (g, x, y, text) => { const w = String(text).length * 6.2 + 9, b = el('g', { class: 'bdg' }, g); el('rect', { x: x - w, y: y - 8, width: w, height: 15, rx: 3 }, b); el('text', { x: x - w / 2, y: y + 3.4 }, b, text); };
  for (const h in MINI_HOST) {
    const b = MINI_HOST[h], g = el('g', { class: `n host ${state[h] || ''} ${pickable ? 'pickable' : ''}`, 'data-id': h }, svg);
    el('rect', { x: b.x, y: b.y, width: b.w, height: b.h, rx: 6 }, g);
    el('text', { class: 'hl', x: h === 'N2' ? b.x + b.w - 10 : b.x + 10, y: b.y + b.h - 9, 'text-anchor': h === 'N2' ? 'end' : 'start' }, g, `${h} · ${NAME[h]}${sub[h] ? ' · ' + sub[h] : ''}`);
    if (badges[h]) badge(g, b.x + b.w - 4, b.y + 2, badges[h]);
  }
  el('line', { class: 'link', x1: 250, y1: 294, x2: 288, y2: 294 }, svg);
  const clip = (a, b, pad) => { const dx = b.x - a.x, dy = b.y - a.y, k = Math.min(dx ? (a.w / 2 + pad) / Math.abs(dx) : Infinity, dy ? (a.h / 2 + pad) / Math.abs(dy) : Infinity); return { x: a.x + dx * k, y: a.y + dy * k }; };
  for (const [u, v, kind] of MINI_EDGES) {
    const p1 = clip(MINI[u], MINI[v], 1), p2 = clip(MINI[v], MINI[u], kind === 'flow' ? 3 : 1);
    el('line', { class: 'e ' + (kind === 'flow' ? '' : kind), x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y }, svg);
    if (kind === 'flow') { const a = Math.atan2(p2.y - p1.y, p2.x - p1.x), z = 7, w = 0.42; el('polygon', { class: 'ah', points: `${p2.x},${p2.y} ${p2.x - z * Math.cos(a - w)},${p2.y - z * Math.sin(a - w)} ${p2.x - z * Math.cos(a + w)},${p2.y - z * Math.sin(a + w)}` }, svg); }
  }
  for (const id in MINI) {
    const n = MINI[id], g = el('g', { class: `n ${n.t === 'Library' ? 'lib' : ''} ${state[id] || ''} ${pickable ? 'pickable' : ''}`, 'data-id': id }, svg), x = n.x - n.w / 2, y = n.y - n.h / 2;
    if (n.t === 'Broker') { const c = 9; el('polygon', { points: [[x + c, y], [x + n.w - c, y], [x + n.w, y + c], [x + n.w, y + n.h - c], [x + n.w - c, y + n.h], [x + c, y + n.h], [x, y + n.h - c], [x, y + c]].map(p => p.join(',')).join(' ') }, g); }
    else el('rect', { x, y, width: n.w, height: n.h, rx: n.t === 'Topic' ? n.h / 2 : 5 }, g);
    el('text', { class: 'nm', x: n.x, y: n.y - 1 }, g, NAME[id]); el('text', { class: 'sb', x: n.x, y: n.y + 11.5 }, g, sub[id] ? `${id} · ${sub[id]}` : id);
    if (badges[id]) badge(g, n.x + n.w / 2 + 5, y - 1, badges[id]);
  }
}
"""


def write_page(template: Path, output: Path, data: dict) -> None:
    """Fill a walkthrough template and write a page that opens directly in a browser."""
    page = template.read_text()
    for marker, value in (("/*BASE_CSS*/", BASE_CSS), ("/*BASE_JS*/", BASE_JS),
                          ("/*DATA*/", json.dumps(data, separators=(",", ":"), default=str))):
        if marker not in page:
            raise ValueError(f"{template.name} has no {marker} marker")
        page = page.replace(marker, value)
    output.write_text(PAGE_HEAD + page)

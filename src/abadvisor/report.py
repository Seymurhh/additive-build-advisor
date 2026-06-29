"""Report rendering: matplotlib figures + a self-contained HTML page.

Turns a pipeline result into (1) a set of PNG figures (orientation DoE, layer
cross-section profile, cost/time breakdown, warpage contributors, and a 3D view
of the oriented part) and (2) a single self-contained HTML report with those
figures embedded as base64, the DfAM and inspection tables color-coded by
severity, and the release-gate banner at the top.

matplotlib is forced onto the non-interactive Agg backend so the report renders
headless (CI, a server, a cron job) with no display.
"""

from __future__ import annotations

import base64
import html as _html
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

_GATE_COLOR = {
    "release_to_build": "#1b7f3b",
    "needs_engineering_review": "#b8860b",
    "redesign_required": "#b22222",
}
_SEV_COLOR = {
    "ok": "#1b7f3b",
    "info": "#2c6fbb",
    "warning": "#b8860b",
    "critical": "#b22222",
}
_MAX_FACETS_3D = 30000


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _fig_orientation(orientation: Dict, path: Path) -> None:
    cands = orientation["candidates"]
    labels = [f"({c.rx_deg:g},{c.ry_deg:g})" for c in cands]
    scores = [c.score for c in cands]
    colors = ["#1b7f3b" if i == 0 else "#9bb7d4" for i in range(len(cands))]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(range(len(cands)), scores, color=colors)
    ax.set_xticks(range(len(cands)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("DoE objective (lower = better)")
    ax.set_xlabel("Candidate orientation (rx, ry) deg")
    ax.set_title("Orientation DoE — green = selected")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fig_layer_profile(sim, path: Path) -> None:
    z = sim.layer_z_mm
    area = sim.layer_area_mm2
    support = sim.support_layer_area_mm2
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.fill_betweenx(z, 0, area, color="#9bb7d4", label="part cross-section")
    if support is not None and float(np.sum(support)) > 0:
        ax.fill_betweenx(z, 0, support, color="#d98c5f", alpha=0.8, label="support")
    ax.set_xlabel("Cross-section area (mm$^2$)")
    ax.set_ylabel("Build height z (mm)")
    ax.set_title("Per-layer cross-section (build simulation)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fig_cost_time(sim, path: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 3.4))
    a1.bar(["build"], [sim.material_cost_usd], label="material", color="#9bb7d4")
    a1.bar(["build"], [sim.machine_cost_usd], bottom=[sim.material_cost_usd],
           label="machine", color="#5f7fa8")
    a1.set_ylabel("Cost (USD)")
    a1.set_title(f"Cost — ${sim.total_cost_usd:.2f}")
    a1.legend(fontsize=8)
    a2.bar(["build"], [sim.deposition_time_h], label="deposition", color="#9bb7d4")
    a2.bar(["build"], [sim.overhead_time_h], bottom=[sim.deposition_time_h],
           label="layer overhead", color="#5f7fa8")
    a2.set_ylabel("Time (h)")
    a2.set_title(f"Time — {sim.total_time_h:.2f} h, {sim.n_layers} layers")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fig_warpage(sim, path: Path) -> None:
    contrib = sim.warpage_contributors
    keys = ["area_gradient", "aspect_ratio", "overhang", "cross_section"]
    vals = [contrib.get(k, 0.0) for k in keys]
    colors = ["#b22222" if v >= 0.5 else "#9bb7d4" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.barh(keys, vals, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalized contribution (0–1)")
    ax.set_title(f"Warpage-risk drivers — index {sim.warpage_index:.0f}/100")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fig_part3d(mesh, path: Path) -> None:
    if mesh.n_facets > _MAX_FACETS_3D:
        return
    fig = plt.figure(figsize=(5.2, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(mesh.triangles, alpha=0.85, facecolor="#9bb7d4", edgecolor="#33506e", linewidths=0.2)
    ax.add_collection3d(coll)
    lo, hi = mesh.bounds
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect(hi - lo)
    except Exception:
        pass
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (build)")
    ax.set_title("Part in build orientation")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def render_figures(result: Dict, outdir: str) -> Dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figs = {
        "orientation": out / "fig_orientation.png",
        "layers": out / "fig_layers.png",
        "cost_time": out / "fig_cost_time.png",
        "warpage": out / "fig_warpage.png",
        "part3d": out / "fig_part3d.png",
    }
    _fig_orientation(result["orientation"], figs["orientation"])
    _fig_layer_profile(result["sim"], figs["layers"])
    _fig_cost_time(result["sim"], figs["cost_time"])
    _fig_warpage(result["sim"], figs["warpage"])
    _fig_part3d(result["oriented_mesh"], figs["part3d"])
    return {k: str(v) for k, v in figs.items() if v.exists()}


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def _img_tag(path: str, embed: bool) -> str:
    if not path or not Path(path).exists():
        return ""
    if embed:
        data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return f'<img src="data:image/png;base64,{data}" />'
    return f'<img src="{Path(path).name}" />'


def _table(headers: List[str], rows: List[List[str]], sev_col: int = None) -> str:
    th = "".join(f"<th>{_html.escape(str(h))}</th>" for h in headers)
    body = []
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            style = ""
            if sev_col is not None and i == sev_col:
                style = f' style="color:#fff;background:{_SEV_COLOR.get(str(c), "#777")};font-weight:600;text-align:center"'
            cells.append(f"<td{style}>{_html.escape(str(c))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table>"


_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;color:#1f2933;background:#f5f7fa}
.wrap{max-width:1000px;margin:0 auto;padding:28px}
h1{font-size:24px;margin:0 0 4px} h2{font-size:18px;margin:28px 0 10px;border-bottom:2px solid #e4e9f0;padding-bottom:6px}
.sub{color:#5b6b7b;margin:0 0 18px}
.banner{padding:16px 18px;border-radius:10px;color:#fff;font-size:18px;font-weight:700;margin:14px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:12px 0}
.card{background:#fff;border:1px solid #e4e9f0;border-radius:10px;padding:12px}
.card .k{color:#5b6b7b;font-size:12px} .card .v{font-size:20px;font-weight:700;margin-top:4px}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e4e9f0;border-radius:8px;overflow:hidden;font-size:13px}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid #eef2f6}
th{background:#eef2f6;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
img{max-width:100%;border:1px solid #e4e9f0;border-radius:8px;background:#fff;margin:8px 0}
.row{display:flex;gap:14px;flex-wrap:wrap} .row>div{flex:1;min-width:320px}
ul{margin:8px 0 8px 18px} code{background:#eef2f6;padding:1px 5px;border-radius:4px}
.foot{color:#7b8794;font-size:12px;margin-top:28px}
"""


def render_html(result: Dict, outdir: str, embed: bool = True, filename: str = "report.html") -> str:
    rec = result["record"]
    figs = render_figures(result, outdir)
    sim = rec["simulation"]
    gate = rec["gate"]
    part = rec["part"]
    proc = rec["process"]

    cards = [
        ("Process", proc["name"]),
        ("Part volume", f"{sim['part_volume_cm3']} cm³"),
        ("Build time", f"{sim['build_time_h']} h"),
        ("Layers", f"{sim['n_layers']}"),
        ("Cost", f"${sim['total_cost_usd']}"),
        ("Warpage index", f"{sim['warpage_index']}/100"),
        ("Volume validation", f"{sim['grid_validation']['volume_error_pct']}%"),
        ("Watertight", "yes" if part["watertight"]["is_watertight"] else "no"),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="k">{_html.escape(k)}</div><div class="v">{_html.escape(str(v))}</div></div>'
        for k, v in cards
    )

    doe = rec["design_decision"]
    orient_rows = [
        [c["index"], f"{c['rx_deg']:g}/{c['ry_deg']:g}", c["height_mm"], c["overhang_area_mm2"],
         c["stability_ratio"], c["score"], "✓" if c == doe["chosen_orientation"] else ""]
        for c in [doe["chosen_orientation"]] + doe["alternatives"]
    ]
    orient_tbl = _table(
        ["#", "rx/ry°", "height mm", "overhang mm²", "stability", "score", "chosen"],
        orient_rows,
    )

    dfam_rows = [[f["check"], f["severity"], f["message"], f["recommendation"]]
                 for f in rec["manufacturability"]["findings"]]
    dfam_tbl = _table(["check", "severity", "finding", "recommendation"], dfam_rows, sev_col=1)

    insp_rows = [[s["feature"], s["characteristic"], s["pass_if"], s["method"],
                  s["equipment"], s["severity"]] for s in rec["inspection_plan"]["steps"]]
    insp_tbl = _table(["feature", "characteristic", "pass if", "method", "equipment", "capability"],
                      insp_rows, sev_col=5)

    reasons = "".join(f"<li>{_html.escape(r)}</li>" for r in gate["reasons"])
    handoff = rec["handoff"]["as_built_context"]

    html_doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Build Advisor — {_html.escape(part['name'])}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>Additive Build Advisor</h1>
<p class="sub">Part <code>{_html.escape(part['name'])}</code> · source <code>{_html.escape(part['source_file'])}</code>
· geometry id <code>{_html.escape(part['geometry_hash'])}</code></p>

<div class="banner" style="background:{_GATE_COLOR.get(gate['decision'], '#555')}">
  Release gate: {_html.escape(gate['decision'].replace('_', ' ').upper())}
  &nbsp;·&nbsp; simulation confidence {gate['confidence']}
</div>
<ul>{reasons}</ul>

<h2>Build summary</h2>
<div class="grid">{cards_html}</div>

<div class="row">
  <div><h2>Orientation (DoE)</h2>{_img_tag(figs.get('orientation'), embed)}{orient_tbl}</div>
  <div><h2>Part in orientation</h2>{_img_tag(figs.get('part3d'), embed)}</div>
</div>

<h2>Build simulation</h2>
<div class="row">
  <div>{_img_tag(figs.get('layers'), embed)}</div>
  <div>{_img_tag(figs.get('cost_time'), embed)}</div>
</div>
{_img_tag(figs.get('warpage'), embed)}

<h2>Manufacturability (DfAM)</h2>
{dfam_tbl}

<h2>Inspection plan</h2>
<p class="sub">Tightest tolerance ±{rec['inspection_plan']['tightest_tolerance_mm']} mm ·
requires CMM: {rec['inspection_plan']['requires_cmm']} · requires CT: {rec['inspection_plan']['requires_ct']}</p>
{insp_tbl}

<h2>Digital-thread hand-off</h2>
<p class="sub">On release, this context is handed to the runtime monitoring twin
(<code>{_html.escape(rec['handoff']['to'])}</code>):</p>
<ul>
  <li>machine_id <code>{_html.escape(handoff['machine_id'])}</code>, part_id <code>{_html.escape(handoff['part_id'])}</code></li>
  <li>operation <code>{_html.escape(handoff['operation'])}</code>, expected {handoff['expected_layers']} layers / {handoff['expected_build_time_h']} h</li>
  <li>signals to watch: {_html.escape(', '.join(handoff['watch']) or 'none flagged')}</li>
</ul>

<p class="foot">Additive Build Advisor · reduced-order estimates from a coarse voxel model, not a substitute for
process-qualified simulation. See REPORT.md for scope and limits.</p>
</div></body></html>"""

    out_path = Path(outdir) / filename
    out_path.write_text(html_doc)
    return str(out_path)

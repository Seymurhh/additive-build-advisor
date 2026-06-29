"""Report rendering: matplotlib figures + a self-contained HTML page.

Turns a pipeline result into (1) a set of PNG figures (orientation screening,
layer cross-section profile, cost/time breakdown, and the FEA distortion field)
and (2) a single self-contained HTML report with those figures embedded as
base64, the DfAM and inspection tables color-coded by severity, and the
release-gate banner at the top.

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
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection  # noqa: E402

# Publication ("Nature"-style) figure defaults: clean sans-serif, light spines,
# no chartjunk, restrained palette, generous DPI.
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "TeX Gyre Heros", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.6,
})

# Restrained palette
_C_PRIMARY = "#2b6cb0"
_C_ACCENT = "#c05621"
_C_SELECT = "#2f855a"
_C_MUTED = "#a0aec0"

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
    labels = [c.label.replace("face ⟂ ", "") for c in cands]
    support = [c.support_volume_mm3 / 1000.0 for c in cands]  # cm^3
    colors = [_C_SELECT if i == 0 else _C_MUTED for i in range(len(cands))]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.bar(range(len(cands)), support, color=colors, width=0.68)
    ax.set_xticks(range(len(cands)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("support material (cm$^3$)")
    ax.set_xlabel("candidate face-down orientation (down-normal)")
    ax.set_title("Orientation screening — selected in green")
    ax.margins(y=0.18)
    # annotate the selected candidate (its bar may be ~0, so point to it)
    ytop = max(support) if max(support) > 0 else 1.0
    ax.annotate("selected", xy=(0, support[0]), xytext=(0, ytop * 0.4),
                ha="center", fontsize=8, color=_C_SELECT, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=_C_SELECT, lw=1.3))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_layer_profile(sim, path: Path) -> None:
    z = sim.layer_z_mm
    area = sim.layer_area_mm2
    support = sim.support_layer_area_mm2
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.fill_betweenx(z, 0, area, color=_C_PRIMARY, alpha=0.85, lw=0, label="part cross-section")
    if support is not None and float(np.sum(support)) > 0:
        ax.fill_betweenx(z, 0, support, color=_C_ACCENT, alpha=0.8, lw=0, label="support")
    ax.set_xlabel("cross-section area (mm$^2$)")
    ax.set_ylabel("build height $z$ (mm)")
    ax.set_title("Per-layer cross-section (build simulation)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_cost_time(sim, path: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 3.4))
    a1.bar(["build"], [sim.material_cost_usd], label="material", color=_C_PRIMARY, width=0.5)
    a1.bar(["build"], [sim.machine_cost_usd], bottom=[sim.material_cost_usd],
           label="machine", color="#1a3f66", width=0.5)
    a1.set_ylabel("cost (USD)")
    a1.set_title(f"Cost — ${sim.total_cost_usd:.2f}")
    a1.legend()
    a2.bar(["build"], [sim.deposition_time_h], label="deposition", color=_C_PRIMARY, width=0.5)
    a2.bar(["build"], [sim.overhead_time_h], bottom=[sim.deposition_time_h],
           label="layer overhead", color="#1a3f66", width=0.5)
    a2.set_ylabel("time (h)")
    a2.set_title(f"Time — {sim.total_time_h:.2f} h, {sim.n_layers} layers")
    a2.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# Local node offsets, and the 6 voxel faces as ordered corner-node offsets.
_FACES = {
    "x-": ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)),
    "x+": ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),
    "y-": ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
    "y+": ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)),
    "z-": ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
    "z+": ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
}
_NEIGHBOR = {"x-": (-1, 0, 0), "x+": (1, 0, 0), "y-": (0, -1, 0),
             "y+": (0, 1, 0), "z-": (0, 0, -1), "z+": (0, 0, 1)}


def _surface_quads(occ):
    """Exposed voxel faces as lists of 4 node-index tuples (the FEA surface mesh)."""
    nx, ny, nz = occ.shape
    quads = []
    ii, jj, kk = np.nonzero(occ)
    for i, j, k in zip(ii, jj, kk):
        for face, (di, dj, dk) in _NEIGHBOR.items():
            ni, nj, nk = i + di, j + dj, k + dk
            inside = (0 <= ni < nx) and (0 <= nj < ny) and (0 <= nk < nz)
            if inside and occ[ni, nj, nk]:
                continue  # interior face, skip
            quads.append([(i + ox, j + oy, k + oz) for (ox, oy, oz) in _FACES[face]])
    return quads


def _fig_distortion(fea, fea_grid, path: Path) -> None:
    """Deformed FEA surface mesh, exaggerated, contour-colored by |u|."""
    occ = fea_grid.occ
    if occ.sum() == 0 or fea.disp_vec is None:
        return
    quads = _surface_quads(occ)
    if not quads:
        return
    p = fea.pitch
    dv = fea.disp_vec
    max_disp = max(fea.max_displacement_mm, 1e-9)
    ext = np.array(occ.shape) * p
    # exaggerate so the peak distortion is ~18% of the largest part dimension
    scale = 0.18 * float(ext.max()) / max_disp

    undeformed, deformed, facemag = [], [], []
    for quad in quads:
        ub, df = [], []
        mags = []
        for (i, j, k) in quad:
            base = np.array([i, j, k], dtype=float) * p
            d = dv[i, j, k]
            ub.append(base)
            df.append(base + d * scale)
            mags.append(float(np.linalg.norm(d)))
        undeformed.append(ub)
        deformed.append(df)
        facemag.append(np.mean(mags))
    facemag = np.array(facemag)

    norm = Normalize(vmin=0.0, vmax=max_disp)
    cmap = plt.get_cmap("inferno")
    fig = plt.figure(figsize=(6.2, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    # faint undeformed reference (build orientation)
    ax.add_collection3d(Line3DCollection(
        [q + [q[0]] for q in undeformed], colors="#c8cdd4", linewidths=0.3))
    poly = Poly3DCollection(deformed, linewidths=0.25, edgecolors="#2d2d2d")
    poly.set_facecolor(cmap(norm(facemag)))
    ax.add_collection3d(poly)

    allpts = np.array([pt for q in deformed for pt in q] + [pt for q in undeformed for pt in q])
    ax.set_xlim(allpts[:, 0].min(), allpts[:, 0].max())
    ax.set_ylim(allpts[:, 1].min(), allpts[:, 1].max())
    ax.set_zlim(allpts[:, 2].min(), allpts[:, 2].max())
    try:
        ax.set_box_aspect((np.ptp(allpts[:, 0]) or 1, np.ptp(allpts[:, 1]) or 1, np.ptp(allpts[:, 2]) or 1))
    except Exception:
        pass
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08)
    cb.set_label("distortion |u| (mm)")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
    ax.set_title(f"Inherent-strain FEA — deformed mesh (×{scale:.0f} exaggerated)\n"
                 f"peak {fea.max_displacement_mm:.3f} mm · {fea.n_elements} elements · "
                 f"{fea.iterations} CG iters", fontsize=9)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_part3d(mesh, path: Path) -> None:
    if mesh.n_facets > _MAX_FACETS_3D:
        return
    fig = plt.figure(figsize=(5.2, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(mesh.triangles, alpha=0.9, facecolor="#9bb7d4", edgecolor="#33506e", linewidths=0.2)
    ax.add_collection3d(coll)
    lo, hi = mesh.bounds
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect(hi - lo)
    except Exception:
        pass
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (build)")
    ax.set_title("Part in build orientation")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_figures(result: Dict, outdir: str) -> Dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figs = {
        "orientation": out / "fig_orientation.png",
        "layers": out / "fig_layers.png",
        "cost_time": out / "fig_cost_time.png",
        "distortion": out / "fig_distortion.png",
        "part3d": out / "fig_part3d.png",
    }
    _fig_orientation(result["orientation"], figs["orientation"])
    _fig_layer_profile(result["sim"], figs["layers"])
    _fig_cost_time(result["sim"], figs["cost_time"])
    _fig_distortion(result["fea"], result["fea_grid"], figs["distortion"])
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
    fea = rec["distortion_fea"]
    gate = rec["gate"]
    part = rec["part"]
    proc = rec["process"]

    cards = [
        ("Process", proc["name"]),
        ("Part volume", f"{sim['part_volume_cm3']} cm³"),
        ("Build time", f"{sim['build_time_h']} h"),
        ("Layers", f"{sim['n_layers']}"),
        ("Cost", f"${sim['total_cost_usd']}"),
        ("FEA max distortion", f"{fea['max_distortion_mm']} mm"),
        ("Peak von Mises", f"{fea['peak_von_mises_mpa']} MPa" if fea["peak_von_mises_mpa"] is not None else "—"),
        ("Volume validation", f"{sim['grid_validation']['volume_error_pct']}%"),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="k">{_html.escape(k)}</div><div class="v">{_html.escape(str(v))}</div></div>'
        for k, v in cards
    )

    doe = rec["design_decision"]
    orient_rows = [
        [c["index"], c["label"].replace("face ⟂ ", ""), c["height_mm"], c["base_contact_mm2"],
         c["support_volume_mm3"], c["score"], "✓" if c == doe["chosen_orientation"] else ""]
        for c in [doe["chosen_orientation"]] + doe["alternatives"]
    ]
    orient_tbl = _table(
        ["#", "down-normal", "height mm", "base contact mm²", "support mm³", "score", "chosen"],
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
  <div><h2>Orientation</h2>{_img_tag(figs.get('orientation'), embed)}{orient_tbl}</div>
  <div><h2>Part in orientation</h2>{_img_tag(figs.get('part3d'), embed)}</div>
</div>

<h2>Build simulation</h2>
<div class="row">
  <div>{_img_tag(figs.get('layers'), embed)}</div>
  <div>{_img_tag(figs.get('cost_time'), embed)}</div>
</div>

<h2>Distortion FEA (inherent-strain method)</h2>
<p class="sub">Target process: metal LPBF · <b>{_html.escape(str(fea.get('applicability', '')))}</b>.
Linear-elastic voxel FEM: {fea['elements']} elements, {fea['dof']} DOF, solved in
{fea['solver_iterations']} CG iterations (converged: {fea['converged']}). Eigenstrain {fea['eigenstrain']}.
Peak distortion <b>{fea['max_distortion_mm']} mm</b>; peak von Mises
{fea['peak_von_mises_mpa']} MPa (linear-elastic, indicative — no plasticity, so it can exceed yield).
Deformed mesh below is exaggerated for visibility.</p>
{_img_tag(figs.get('distortion'), embed)}

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
  <li>signals to watch: {_html.escape(', '.join(handoff['watch']))}</li>
</ul>

<p class="foot">Additive Build Advisor · reduced-order estimates plus an inherent-strain FEA;
representative process constants, not a melt-pool-calibrated thermo-mechanical solve. See REPORT.md for scope and limits.</p>
</div></body></html>"""

    out_path = Path(outdir) / filename
    out_path.write_text(html_doc)
    return str(out_path)

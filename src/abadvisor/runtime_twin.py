"""Runtime FFF print-monitoring digital twin (the back half of the thread).

Where the advisor ends -- a released build with a machine-readable hand-off
record -- this picks up. Once a part is on the printer, this twin models the
*print in progress*: it synthesizes the sensor streams an instrumented FFF
machine produces, compares each against the expected process envelope, flags
deviation windows, tracks a smoothed health score, and issues a
**verify-before-act** recommendation that escalates as evidence accrues.

It is deliberately the additive analog of a runtime CNC process monitor, but for
FFF: instead of spindle load and chatter it watches hotend and bed temperature,
extrusion flow, frame vibration, and corner-lift off the bed (the warp the
advisor's FEA predicted). A small library of injected faults exercises the
realistic failure modes of an FFF print:

* ``warp_adhesion``   -- the part contracts and lifts off the bed (corner-lift)
* ``under_extrusion`` -- flow falls (a partial clog or grinding filament)
* ``layer_shift``     -- a mechanical skip (vibration spike, shifted layer)
* ``thermal_drift``   -- the hotend drifts out of band (PID / thermistor fault)
* ``sensor_dropout``  -- a sensor flatlines, so the twin *refuses* to act

The same verify-before-act discipline as the advisor's release gate: on a sensor
dropout it explicitly declines to recommend a parameter change rather than acting
on incomplete data.

The simulation is fully deterministic -- seeded from the part's geometry hash and
the scenario -- so the same part + scenario always renders the same dashboard.
No dependency beyond numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# scenario key -> (label, one-line description, injected-fault channel)
SCENARIOS: Dict[str, tuple] = {
    "nominal": ("Nominal print", "Every channel stays inside its envelope — a clean build.", None),
    "warp_adhesion": ("Bed-adhesion loss (warping)",
                      "The part contracts as it cools and its corners lift off the bed.", "corner_lift_um"),
    "under_extrusion": ("Under-extrusion / clog",
                        "Extrusion flow falls — a partial nozzle clog or grinding filament.", "flow_pct"),
    "layer_shift": ("Layer shift",
                    "A mechanical skip — a vibration spike and a shifted layer.", "vibration_g"),
    "thermal_drift": ("Hotend thermal drift",
                      "Hotend temperature drifts out of band — a PID or thermistor fault.", "hotend_c"),
    "sensor_dropout": ("Sensor dropout",
                       "A sensor flatlines; the twin refuses to act on incomplete data.", "dropout"),
    "stress_demo": ("Multi-fault stress test",
                    "A staged sequence — under-extrusion, a layer shift, a sensor dropout (held), then "
                    "warping — that exercises the whole monitor at once.", "multi"),
}

# representative FFF nozzle / bed setpoints (deg C) by material
_MATERIAL_TEMPS = {"PLA": (210.0, 60.0), "ABS": (245.0, 100.0), "PETG": (235.0, 80.0),
                   "PA12": (0.0, 0.0), "Photopolymer resin": (0.0, 0.0)}

_SEV_RANK = {"ok": 0, "info": 1, "warning": 2, "critical": 3}


@dataclass
class TwinResult:
    scenario: str
    label: str
    description: str
    layers: np.ndarray                     # (n,) layer index 1..n
    n_layers: int
    sensors: Dict[str, dict]               # key -> {label, unit, actual, lo, hi, target}
    health: np.ndarray                     # (n,) smoothed 0..1
    deviation: np.ndarray                  # (n,) worst normalized band-exceedance
    data_ok: np.ndarray                    # (n,) bool: all sensors reporting
    anomalies: List[dict]                  # {sensor, label, start, end, severity, message}
    recs: List[dict]                       # per-layer recommendation
    watch: List[str]

    # ---- convenience --------------------------------------------------
    def snapshot(self, k: int) -> dict:
        """State 'as of' layer index ``k`` (0-based), for the live dashboard."""
        k = int(np.clip(k, 0, self.n_layers - 1))
        active = [a for a in self.anomalies if a["start"] <= k <= a["end"]]
        return {
            "layer": k + 1,
            "progress": (k + 1) / self.n_layers,
            "health": float(self.health[k]),
            "active": active,
            "rec": self.recs[k],
        }

    @property
    def worst_severity(self) -> str:
        if not self.anomalies:
            return "ok"
        return max((a["severity"] for a in self.anomalies), key=lambda s: _SEV_RANK[s])

    @property
    def focus_layer(self) -> int:
        """The most informative layer to open the dashboard on (0-based)."""
        if not self.data_ok.all():                       # a dropout: sit inside its window
            miss = np.nonzero(~self.data_ok)[0]
            return int(miss[len(miss) // 2])
        if float(self.deviation.max()) > 0.12:           # otherwise the worst deviation
            return int(np.argmax(self.deviation))
        return self.n_layers - 1                         # nominal: the finished print

    @property
    def final_rec(self) -> dict:
        return self.recs[-1]


# --------------------------------------------------------------------------- #
# Sensor envelopes
# --------------------------------------------------------------------------- #
def _sensor_specs(t_noz: float, t_bed: float, frac: np.ndarray):
    """Return per-sensor (label, unit, target, lo-array, hi-array, band-width)."""
    n = frac.shape[0]
    ones = np.ones(n)
    return {
        "hotend_c": ("Hotend temperature", "°C", t_noz, (t_noz - 4) * ones, (t_noz + 4) * ones, 8.0),
        "bed_c": ("Bed temperature", "°C", t_bed, (t_bed - 3.5) * ones, (t_bed + 3.5) * ones, 7.0),
        "flow_pct": ("Extrusion flow", "%", 100.0, 91.0 * ones, 109.0 * ones, 18.0),
        "vibration_g": ("Frame vibration", "g", 0.05, 0.0 * ones, 0.13 * ones, 0.13),
        # allowed corner-lift grows a little with height; warp breaches it
        "corner_lift_um": ("Corner lift (adhesion)", "µm", 0.0, 0.0 * ones, 45.0 + 55.0 * frac, 100.0),
    }


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def simulate_runtime(record: Dict, scenario: str = "warp_adhesion",
                     seed: Optional[int] = None) -> TwinResult:
    """Simulate an instrumented FFF print of a released part under ``scenario``."""
    if scenario not in SCENARIOS:
        raise KeyError(f"Unknown scenario {scenario!r}. Available: {', '.join(SCENARIOS)}")
    label, desc, _ = SCENARIOS[scenario]

    ctx = record.get("handoff", {}).get("as_built_context", {})
    n = int(np.clip(int(ctx.get("expected_layers", 120)), 24, 180))
    material = record.get("process", {}).get("material", "PLA")
    t_noz, t_bed = _MATERIAL_TEMPS.get(material, (210.0, 60.0))
    if t_noz == 0.0:  # non-FFF process: still show an FFF-style print monitor
        t_noz, t_bed = 210.0, 60.0
    warp_mm = float(record.get("distortion_fea", {}).get("max_distortion_mm", 0.2) or 0.2)
    watch = list(ctx.get("watch", []))

    ghash = record.get("part", {}).get("geometry_hash", "0") or "0"
    if seed is None:
        seed = (int(ghash[:8], 16) + list(SCENARIOS).index(scenario) * 97) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    layers = np.arange(1, n + 1)
    frac = layers / n
    specs = _sensor_specs(t_noz, t_bed, frac)

    # ---- nominal streams ---------------------------------------------
    hotend = t_noz + rng.normal(0, 0.5, n)
    bed = t_bed + rng.normal(0, 0.4, n)
    flow = 100.0 + rng.normal(0, 1.2, n)
    vib = np.abs(0.05 + rng.normal(0, 0.008, n))
    corner = 6.0 + 14.0 * frac + np.abs(rng.normal(0, 1.5, n))  # gentle, within envelope

    data_nan = np.zeros((5, n), dtype=bool)  # tracks the dropped sensor for data_ok
    onset = int(0.55 * n)

    # ---- inject the fault --------------------------------------------
    if scenario == "warp_adhesion":
        # corner-lift ramps beyond the allowed band; magnitude scales with the
        # advisor's predicted warpage. also a small vibration rise as it lifts.
        peak = max(120.0, warp_mm * 1000.0 * 0.9)
        ramp = np.clip((frac - 0.5) / 0.5, 0, 1) ** 1.6
        corner = corner + peak * ramp
        vib = vib + 0.04 * ramp
    elif scenario == "under_extrusion":
        w = np.clip((layers - onset) / max(1, int(0.12 * n)), 0, 1)
        w = w * np.clip((int(0.9 * n) - layers) / max(1, int(0.1 * n)), 0, 1)  # taper at end
        flow = flow - 33.0 * w
    elif scenario == "layer_shift":
        k0 = int(0.5 * n)
        for d, amp in ((0, 0.55), (1, 0.28), (2, 0.12)):
            if k0 + d < n:
                vib[k0 + d] += amp
    elif scenario == "thermal_drift":
        drift = np.clip((layers - onset) / max(1, (n - onset)), 0, 1)
        hotend = hotend + 16.0 * drift
    elif scenario == "sensor_dropout":
        w0, w1 = int(0.45 * n), int(0.72 * n)
        vib[w0:w1] = np.nan          # the vibration channel flatlines / drops out
        data_nan[3, w0:w1] = True
    elif scenario == "stress_demo":  # a staged sequence of faults
        a0, a1 = int(0.16 * n), int(0.30 * n)          # 1) under-extrusion
        w = (np.clip((layers - a0) / max(1, int(0.05 * n)), 0, 1)
             * np.clip((a1 - layers) / max(1, int(0.05 * n)), 0, 1))
        flow = flow - 32.0 * w
        ks = int(0.42 * n)                              # 2) layer shift
        for d, amp in ((0, 0.5), (1, 0.24)):
            if ks + d < n:
                vib[ks + d] += amp
        d0, d1 = int(0.55 * n), int(0.68 * n)          # 3) sensor dropout (held)
        vib[d0:d1] = np.nan
        data_nan[3, d0:d1] = True
        ramp = np.clip((frac - 0.78) / 0.22, 0, 1) ** 1.5   # 4) warping at the end
        corner = corner + max(120.0, warp_mm * 1000.0 * 0.9) * ramp

    actual = {"hotend_c": hotend, "bed_c": bed, "flow_pct": flow,
              "vibration_g": vib, "corner_lift_um": corner}
    keys = list(actual.keys())

    # ---- assemble sensor records + per-step deviation -----------------
    sensors: Dict[str, dict] = {}
    dev = np.zeros((len(keys), n))
    for si, key in enumerate(keys):
        lbl, unit, target, lo, hi, width = specs[key]
        a = actual[key]
        over = np.maximum(a - hi, lo - a)
        over = np.where(np.isnan(a), 0.0, over)
        dev[si] = np.clip(over / width, 0.0, None)
        sensors[key] = {"label": lbl, "unit": unit, "actual": a, "lo": lo, "hi": hi,
                        "target": target}

    data_ok = ~data_nan.any(axis=0)
    worst_dev = dev.max(axis=0)

    # ---- smoothed health score (EMA) ---------------------------------
    raw = np.clip(1.0 - worst_dev, 0.0, 1.0)
    raw = np.where(data_ok, raw, np.minimum(raw, 0.8))  # missing data caps confidence
    health = np.empty(n)
    h = 1.0
    for i in range(n):
        h = 0.7 * h + 0.3 * raw[i]
        health[i] = h

    # ---- anomaly windows (per sensor) --------------------------------
    anomalies: List[dict] = []
    for si, key in enumerate(keys):
        flagged = dev[si] > 0.12
        i = 0
        while i < n:
            if flagged[i]:
                j = i
                while j + 1 < n and flagged[j + 1]:
                    j += 1
                peak = float(dev[si, i:j + 1].max())
                dur = j - i + 1
                sev = ("critical" if peak > 0.6 or (peak > 0.3 and dur >= 4)
                       else "warning" if peak > 0.25 or dur >= 3 else "info")
                anomalies.append({
                    "sensor": key, "label": specs[key][0], "start": i, "end": j,
                    "severity": sev, "message": _anomaly_msg(key, sev),
                })
                i = j + 1
            else:
                i += 1

    # ---- per-layer verify-before-act recommendation ------------------
    recs = [_recommend(k, keys, dev, data_ok, data_nan, specs) for k in range(n)]

    return TwinResult(
        scenario=scenario, label=label, description=desc, layers=layers, n_layers=n,
        sensors=sensors, health=health, deviation=worst_dev, data_ok=data_ok,
        anomalies=anomalies, recs=recs, watch=watch,
    )


# --------------------------------------------------------------------------- #
# Messaging / recommendation
# --------------------------------------------------------------------------- #
_ACTIONS = {
    "corner_lift_um": ("Pause & intervene",
                       "Corner-lift indicates the part is warping off the bed. Pause, add a brim/raft, "
                       "raise bed temperature, and slow part cooling before it detaches."),
    "flow_pct": ("Pause & intervene",
                 "Extrusion flow is below envelope — a partial clog or filament grind. Pause, purge the "
                 "nozzle, and check feeder tension / filament path."),
    "vibration_g": ("Abort recommended",
                    "A vibration spike consistent with a layer shift. Stop before more layers deposit; "
                    "check belt tension and stepper current."),
    "hotend_c": ("Hold & verify",
                 "Hotend temperature is drifting out of band. Hold and verify the thermistor and PID "
                 "tune before trusting deposition temperature."),
    "bed_c": ("Hold & verify", "Bed temperature is out of band; verify the bed heater and sensor."),
}


def _anomaly_msg(key: str, sev: str) -> str:
    base = {
        "corner_lift_um": "Corner-lift / bed-adhesion loss",
        "flow_pct": "Under-extrusion (flow below envelope)",
        "vibration_g": "Vibration spike (possible layer shift)",
        "hotend_c": "Hotend temperature out of band",
        "bed_c": "Bed temperature out of band",
    }.get(key, "Out-of-envelope deviation")
    return f"{base} — {sev}"


def _recommend(k, keys, dev, data_ok, data_nan, specs) -> dict:
    """Verify-before-act recommendation as of layer index ``k``."""
    # 1) refuse on missing data — do not act on an incomplete picture
    if not data_ok[k]:
        dropped = [specs[keys[si]][0] for si in range(len(keys)) if data_nan[si, k]]
        return {
            "status": "refuse", "color": "info",
            "title": "Holding — sensor dropout",
            "detail": (f"{', '.join(dropped)} not reporting. The twin will not recommend a parameter "
                       "change on incomplete data — restore the sensor, then re-evaluate."),
            "confidence": 0.0, "refused": True,
        }

    # 2) worst current channel + how long it has been out
    di = dev[:, k]
    si = int(np.argmax(di))
    d = float(di[si])
    key = keys[si]
    run = 0
    j = k
    while j >= 0 and dev[si, j] > 0.12:
        run += 1
        j -= 1

    if d <= 0.12:
        return {"status": "continue", "color": "ok", "title": "Continue — within envelope",
                "detail": "All channels are tracking their expected process models.",
                "confidence": round(float(0.7 + 0.3 * min(1.0, k / 20)), 2), "refused": False}
    if d < 0.30 and run < 3:
        return {"status": "watch", "color": "warning",
                "title": f"Caution — watching {specs[key][0].lower()}",
                "detail": "An emerging deviation is being tracked; not yet enough evidence to act.",
                "confidence": round(0.4 + 0.4 * d, 2), "refused": False}
    title, detail = _ACTIONS.get(key, ("Intervene", "A sustained deviation is out of envelope."))
    color = "critical" if (d > 0.6 or run >= 4) else "warning"
    conf = round(min(0.98, 0.6 + 0.25 * d + 0.05 * run), 2)
    return {"status": "intervene", "color": color, "title": title, "detail": detail,
            "confidence": conf, "refused": False}

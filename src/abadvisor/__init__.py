"""Additive Build Advisor.

A small, readable design-to-inspection digital thread for additive
manufacturing. It takes a part geometry (STL), recovers a clean mesh, chooses a
build orientation through a design-of-experiments sweep, simulates the build
(layers, support, time, material, cost, and a reduced-order warpage-risk
index), runs Design-for-Additive-Manufacturing checks, generates an inspection
plan from the part's tolerances, and assembles a single machine-readable
digital-thread record with an explicit release gate.

The package is intentionally compact and built from first principles on top of
numpy so the engineering decisions are easy to read and defend, not hidden
behind a CAD kernel. See README.md and REPORT.md for scope and honest limits.
"""

__version__ = "0.1.0"

# Convenience top-level entry point. Guarded so the subpackages remain
# importable even while the package is being assembled.
try:  # pragma: no cover - trivial import shim
    from .pipeline import advise
except ImportError:  # pragma: no cover
    advise = None

__all__ = ["advise", "__version__"]

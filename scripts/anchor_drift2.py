"""Compatibility entry point for candidate-window anchor drift measurement.

Both drift commands use inclusive candidate labels: source-window index 0 is
window-lines[0]. See anchor_drift.py; reconstructed paper text is never used.
"""
from anchor_drift import main

if __name__ == "__main__":
    raise SystemExit(main())

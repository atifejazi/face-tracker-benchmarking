# Lab note: ns017 MICA/SMIRK GA regression (2026-10-04)

**Broken run:** cancel-camera MICA default + SMIRK `--focal 1200 --camera-z 1.0` + Rx(π) on rotation only → MICA PSNR ~20.9, SMIRK ~18.7. VHAP (~29.6) was fine.

**Fix:** MICA export default = legacy moving-camera; SMIRK inherits MICA K/z, no Rx by default (`--opencv-y-flip` for full R&t if needed).

See `AGENT_HANDOFF_NS017_REGRESSION.md` for full diagnosis and re-run commands.

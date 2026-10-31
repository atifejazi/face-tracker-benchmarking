# Mesh and camera checks

## SMIRK export

The default `export_smirk_to_ga.py` mode is `--projection-mode smirk`.
SMIRK maps FLAME coordinates to crop pixels as
`112*(1+s*(X+tx)), 112*(1-s*(Y+ty))`, then applies the inverse crop.
The GA export uses a fixed OpenCV camera rotation `diag(1,-1,-1)` and
converts all three translation components to world coordinates. It retains the
native FLAME root and jaw pose; it does not bake a camera flip into LBS.
GA's symmetric FOV projection uses centered square pixels, rather than the
JSON principal-point metadata. Weak-to-perspective conversion is approximate
away from the reference Z=0 plane. SMIRK eyelid offsets are not exported.

The historical exporter inverted the mesh. Its camera-only flip retained
incorrect translation, so that flag did not validate the complete conversion.
`--projection-mode legacy` reproduces historical behavior; the historical flip
flags require that explicit mode. RGB images should never be flipped.

Run after export, before training:

```bash
python "$REPO_ROOT/scripts/diagnose_ga_mesh_overlay.py" \
  --dataset "$TGT" --ga-root "$GA_ROOT" --track-params "$TRACK_PT" \
  --frames all --out_dir "$WORKDIR/mesh_diag"
```

The tool decodes GA's actual FlameHead LBS geometry. Its panels show the
exported mesh and native weak-camera reference, with eye, nose, brow and chin
labels. JSON reports and a nonzero exit expose inverted/hidden faces or mean
native-reference error above 8 pixels. This measures export consistency, not
accuracy against ground truth. The sentence runner checks sampled frames and
stops before training on a failed check.

CPU validation covered 640 frames across seven saved clips and four subjects:
all were upright, with per-clip mean native-reference differences of 1.70–4.49
pixels. Fresh GA training PSNR for the corrected export is not yet validated.
Use new dataset/model names and retrain when changing mesh/camera conventions;
old learned Gaussians cannot validate the corrected geometry. Good image PSNR
alone does not establish correct underlying mesh geometry.

`render_mesh_overlay_sequence.py` exports a labeled sequence on source images.
`make_tracker_wireframe_sbs.py` overlays the exported tracker meshes on GA renders;
these are dataset meshes, not a reconstruction of fitted checkpoint geometry.
Both share the GA LBS and rasterizer projection implementation.

## MICA export

The default remains the original moving-camera export. The opt-in
`--camera-mode relative_cancel` keeps the camera fixed while retaining head
motion in FLAME. Its translation compensates for the shaped FLAME root joint
(J0), so root LBS matches the original camera projection. Check a paired export:

```bash
python "$REPO_ROOT/scripts/verify_mica_camera_equiv.py" \
  --legacy /path/to/legacy --relative /path/to/relative --ga-root "$GA_ROOT"
```

Apply the dependency patches before using the fixed-camera path. Follow the
regularized training schedule in README; compare fresh runs with the same
settings and targets. Local experiment-specific scripts and models are not
included in this distribution.

## Regression tests

In the gaussian-avatars environment, from this repository:

```bash
python -m unittest discover -s tests -v
```

The camera tests do not need CUDA or model weights. End-to-end tracking and GA
training require the separate dependencies, assets and GPU environment.

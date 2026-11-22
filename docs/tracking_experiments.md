# Tracking error experiments

Run the commands from this repository. Use the existing GaussianAvatars environment for FLAME, CUDA and image operations. The MultiREX import and grouping commands only use Python's standard library.

For a cross-tracker avatar comparison, use the [common-canvas protocol](common_canvas.md) to share source timing, RGB, masks and evaluation splits.

## 1. Audit and group

Import the per-frame CSV produced by `scripts/static_geometry/run_multirex_eval_subsampled.py` for each tracker. Supply a protocol name that records the frame sampling and tracker conditions. These errors use reference neutral shape and region-level rigid alignment. They measure expression deformation under that protocol.

```bash
python scripts/audit_tracking.py multirex --protocol front_stride8 \
  --metrics mica "$MULTIREX_RESULTS/eval/mica/metrics.csv" \
  --metrics smirk "$MULTIREX_RESULTS/eval/smirk/metrics.csv" \
  --metrics vhap "$MULTIREX_RESULTS/eval/vhap/metrics.csv" \
  --output runs/tracking_audit/multirex.csv
python scripts/audit_tracking.py summarize --metrics runs/tracking_audit/multirex.csv \
  --group-by camera glasses facial_hair motion_speed --output runs/tracking_audit/groups.csv
```

To supply labels, pass `--clip-labels labels.csv` with keys `dataset,subject,sequence,camera`. Copy keys from the normalized measurements, preserving leading zeros. For time-varying labels, pass `--frame-labels frame_labels.csv` with the additional `frame` key. Frame labels override clip labels when supplied. Supported fields are `glasses,facial_hair,occlusion,expression_intensity,mouth_motion,head_pose,motion_speed`. Use labels from the reference or manual annotation, with categories defined before comparing trackers. Missing label values become `unknown`. Each summary row gives identity, clip and observation counts. The mean weights frames equally within a clip, clips equally within an identity and identities equally within a group. Compare trackers on verified shared observations before interpreting differences. Regions, datasets, protocols, motion/view splits and units are always kept distinct.

For a GaussianAvatars export, generate a manifest for a single render split. The split refers to the JSON file. Declare seen/unseen motion and view from run provenance. A contiguous test block in a training sentence is still the same motion sequence.

```bash
python scripts/audit_tracking.py ga-manifest --source "$DATASET" --split test \
  --dataset NeRSemble --subject 018 --sequence SEN-01 \
  --motion-split seen --view-split seen --capture monocular \
  --output runs/tracking_audit/test_manifest.csv
python scripts/audit_tracking.py score-rgb --manifest runs/tracking_audit/test_manifest.csv \
  --renders "$RENDER_DIR" --references "$GA_REFERENCE_DIR" --tracker mica \
  --protocol matched_test_frames --output runs/tracking_audit/test_metrics.csv
```

Use GA with `--eval` for a split-specific training/render run and camera filtering disabled. The manifest maps GA's sequential PNG names to export timesteps and camera IDs. Reference files should be GA's paired `gt/` images or independent references rasterized to the same pixel grid and named by the manifest. The scorer requires exact file sets and dimensions. Original export indices are recorded separately from GA indices and require a verified source time mapping before making audio claims.

## 2. Run a perturbation sweep

```bash
python scripts/perturb_avatar.py --ga-root "$GA_ROOT" --dataset "$DATASET" \
  --checkpoint "$MODEL/point_cloud/iteration_25000" \
  --dataset-name NeRSemble --subject 018 --sequence SEN-01 --tracker mica \
  --motion-split seen --view-split seen --family lag --doses 40 80 --fps 25 \
  --render --resolution 2 --sh-degree 0 --white-background \
  --output runs/tracking_audit/ns018_lag
```

Each sweep includes a zero-dose control. The families and proposed pilot doses are `shape` and `expression` with additive 0.25 and 0.75 on one FLAME coefficient, `jaw` with 1 and 3 degrees about a local axis, `lower_face` with 1 and 3 mm neutral-RMS downward displacement inside a smooth supported chin-region mask, and `lag` with 40 and 80 ms applied jointly to jaw and expression. `--component` selects the coefficient or axis (0/1/2 for X/Y/Z). Use the same FLAME version, mesh topology, pose convention and world scale as the checkpoint.

For lag, provide the verified exported constant frame rate. The tool checks contiguous exported source indices. It trims the evaluated interval identically for all doses using the largest lag. Expression is interpolated linearly and jaw rotations use SLERP. RGB, camera and timestamps stay on the original grid. The pivot correction keeps the rigid root transform fixed when shape, expression or a static offset changes the root joint. Mesh displacement is reported over the lower-face support, lips or entire mesh, depending on the family. Each frame also records distance between the inner-lip landmarks normalized by interocular distance, and error relative to the zero-dose mesh.

Omit `--render` to validate geometry on CPU. Render mode uses the checkpoint's learned splats and the stock GA CUDA rasterizer. Set render resolution, SH degree and background to match the model. Each per-dose directory records the derived FLAME parameters and geometry. Render mode also saves exact paired PNGs and full-frame/mouth PSNR. The mouth window and, for the local offset family, the chin window follow the zero-dose mesh and are reused for all doses. The input track provides the reference for the geometry measurements. The RGB images are observed targets. Differences between doses estimate response to the injected change.

For unseen motion, supply `--drive-dataset` pointing to a separate GA export for the same subject with the same FLAME basis and world scale. The checkpoint supplies shape and static offset. The drive export supplies expression, poses, translation and cameras. Check projections on the new clip, especially if the tracking crop or fitted camera gauge changed. `experiment.json` records the subset and per-frame paths so runs can be audited.

For matched retraining, add `--export-training-datasets` to a source-sequence sweep. Each dose directory then contains a GA dataset with the same RGB and split assignment. Lag variants share the trimmed valid interval and are reindexed. Train each dataset in its own model folder using `--eval --bind_to_mesh --not_finetune_flame_params` and the same budget, seed, loss, camera and optimizer settings. If the upstream visual logger divides by zero on a small split, schedule its `--test_iterations` after the last training iteration and evaluate the saved checkpoint separately. The per-dose data and renders belong under the ignored `runs/` directory. For compensation claims, compare the arms at identical drive geometry and check a held-out sequence or view. A comparison of each model driven by its own fit combines training and driving effects.

For a checkpoint-by-drive comparison on the same camera/timestamp grid, pass `--drive-state` with the desired full FLAME sidecar. This explicitly replaces the supporting mesh state while preserving the loaded Gaussian weights and bindings. Omit `--doses` for a single control pass. Compare model checkpoints on the same drive sidecar before comparing across drives.

For geometry checks or training export directly from a GA track, omit `--checkpoint` and use `--dataset` as the reference. This path reads the original exported FLAME parameters before any GaussianAvatars mesh fine-tuning. Rendering and alternate drives require a checkpoint.

## 3. Reproduce the expanded NeRSemble study

Use the GaussianAvatars environment set up by [SETUP.md](../SETUP.md), with the repository's GA patches applied. Start with two monocular GA exports per identity on the same physical camera and a verified 25 fps time grid. The source is SEN-01 and must have a nonempty train and test split. The drive is SEN-05. Each export needs a shared identity, compatible FLAME topology and units, three `transforms_<split>.json` files, and the RGB and mesh paths named in those files. Verify the crop, projection and timing for both clips before comparing render quality. For a fixed-camera export from an existing Metrical Tracker run, use the benchmark exporter with `--camera-mode relative_cancel`.

Run from the benchmark checkout after activating the GA environment. Adapt the export locations to your machine. The runner starts from exported FLAME, so it needs no previously trained avatar.

```bash
python scripts/tracking_experiments/run_expanded.py \
  --ga-root "$GA_ROOT" --output runs/expanded_verification \
  --case "$SUBJECT" "$SEN01_EXPORT" "$SEN05_EXPORT"
```

Repeat `--case` for each eligible identity. The defaults reproduce the 3,500-step recipe and 32-frame sensitivity sample. Use `--iterations 2 --frames 4` and one case for a short smoke test. Use a fresh output directory for each run. Set `--port` if GA's network port is already occupied. The runner writes per-stage commands and logs, a run configuration, and a completion marker after the final render. It produces two checkpoints per identity, three frozen-avatar sweeps on each sequence, and the crossed clean/biased training and drive conditions. It renders all SEN-01 test frames for the crossed same-sentence comparison. The unseen comparison uses the same sampled frames as the clean model's unseen chin sweep. Model weights, frames and local run artifacts belong in the ignored run directory.

### Replicate the matched study across seeds

For explicit seeds, apply the Python-only GA patch after the usual dependency setup. The original implicit seed remains 0.

```bash
seed_patch="$PWD/patches/ga-training-seed.patch"
if ! git -C "$GA_ROOT" apply --reverse --check "$seed_patch"; then
  git -C "$GA_ROOT" apply "$seed_patch"
fi

for seed in 0 1; do
  python scripts/tracking_experiments/run_expanded.py \
    --ga-root "$GA_ROOT" --output "runs/${SUBJECT}_seed_${seed}" \
    --case "$SUBJECT" "$SEN01_EXPORT" "$SEN05_EXPORT" --seed "$seed"
done
```

Use the same case list, frame sample and training schedule for each seed. Within each run, the clean and biased avatars receive the same seed. GA seeds Python, NumPy and PyTorch, affecting initial Gaussian colors, training-frame order and random split samples during densification. The FLAME perturbations, tracker outputs and evaluation frames are deterministic inputs to this comparison. The seed is recorded in `run_config.json` and the trained models' `cfg_args`. Compare the paired retraining gains and train-by-drive interactions per identity across seeds. Use fresh training runs for this comparison. CUDA operations can introduce small numerical differences on a same-seed replay, and checkpoint resumes do not restore RNG state.

With the GA environment active, run the split/lag validation and rebuild the numerical summaries.

```bash
python scripts/tracking_experiments/validate_run.py runs/expanded_verification
python scripts/tracking_experiments/analyze.py runs/expanded_verification
```

The analysis checks frame keys and matching reference pixels, summarizes dose effects by identity, recomputes the four-condition chin metric with a single ROI, and reads the learned splat offsets from each checkpoint. To analyze the original 3,500-step archive, supply its path and `--ga-root "$GA_ROOT"`. Summary CSVs are written to the supplied run directory.

Use the existing `tracker` environment with OpenCV and the repository's MediaPipe Face Mesh version for the rendered lip-aperture check.

```bash
conda activate tracker
python scripts/tracking_experiments/measure_aperture.py runs/expanded_verification
```

Face Mesh detects the inner-lip and outer-eye corners in each reference and render. The error is the absolute difference in inner-lip distance divided by the reference interocular distance. A frame must have valid landmarks in the reference and every dose to enter a paired comparison. The per-frame CSV preserves detector failures and the summary records paired coverage. The metric measures image-based articulation on the selected frames.

See the [perturbation method](tracking_experiment_method.md) for the comparison design. The analysis writes CSVs in the local run directory.

## 4. Run the locked perturbation protocol

Use the shared canvas and both optional GA patches, [common-canvas cameras](common_canvas.md) and [training seeds](#replicate-the-matched-study-across-seeds). The protocol uses SEN-01 training and its chronological test block, SEN-09 for primary unseen-motion evaluation, and SEN-05 for replication. It crosses clean, 3 mm chin-bias, and 80 ms lag training with their respective clean and biased drives. The other perturbation families are included in the fixed-avatar dose sweeps.

Prepare a clip under `INPUTS/SUBJECT/SEN-XX` with `prepare_protocol_clip.py` in the `tracker` environment. Supply the physical source video, Metrical Tracker actor and checkpoint paths, start frame count, cropped alpha directory, matching GA FLAME asset, and the full-frame prefit video and alpha directory. The helper verifies the time grid against decoded RGB, Metrical Tracker input crops and prefit JPEGs before exporting. Include `--vhap` for 017 and 018 on SEN-01 and SEN-09. Then run SMIRK on those clips' common tracking images and check its full posed geometry with the [common-canvas procedure](common_canvas.md).

Before the full lock, run `lock`, `train` and `render` with `--smoke` and a separate output path. This is a two-step experiment on three SEN-01 validation frames. Validate the locked inputs and record detector provenance before full training:

```bash
python scripts/tracking_experiments/run_protocol.py lock \
  --ga-root "$GA_ROOT" --inputs "$INPUTS" --output "$RUN"
conda run -n tracker python scripts/tracking_experiments/score_protocol.py reference "$RUN"
python scripts/tracking_experiments/run_protocol.py train \
  --ga-root "$GA_ROOT" --inputs "$INPUTS" --output "$RUN"
python scripts/tracking_experiments/run_protocol.py render \
  --ga-root "$GA_ROOT" --inputs "$INPUTS" --output "$RUN"
conda run -n tracker python scripts/tracking_experiments/score_protocol.py landmarks "$RUN"
python scripts/tracking_experiments/score_protocol.py rgb "$RUN"
python scripts/tracking_experiments/score_protocol.py analyze "$RUN"
```

The lock stage accepts `--subjects` for independent batches of preregistered identities. Keep settings and code hashes identical. Aggregate disjoint completed batches with `score_protocol.py analyze "$FIRST_BATCH" --include "$SECOND_BATCH" --results "$SUMMARY"`. The aggregator requires identical detector and LPIPS provenance.

The lock records split support, code and input hashes, mesh geometry doses, shared scored target frames, and fixed chin boxes. The first two frames provide lag history and are excluded from training and scoring across all arms. The same clean model serves the two retraining comparisons per identity and seed. Renders use the frozen checkpoint and a common drive state per matched cell. The score stage uses shared reference-derived face and mouth boxes, AlexNet LPIPS on the full raster and 96-pixel face crops, regional PSNR and the aperture proxy. Summaries preserve paired frame counts, average across seeds within each identity and include leave-one-identity-out medians. The study and local outputs are separate from the historical fixture runner above.

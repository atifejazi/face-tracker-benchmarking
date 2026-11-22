# Common-canvas tracker comparison

Use this protocol to compare what native monocular tracker outputs yield under the same GaussianAvatars training and scoring inputs. The existing perturbation experiments estimate within-track responses. Their matched arms already share the images, crops, cameras and split used for each contrast. Historical scores from the native MICA, SMIRK and VHAP pipelines describe whole systems with different preprocessing and avatar settings.

## 1. Verify source frames and split membership

Decode the original physical camera once, on the intended time grid. Supply one full-frame alpha sequence computed before tracker fitting or mesh masking. For the demonstrated NeRSemble protocol, the source was a single camera and the prefit full-resolution RVM alphas came from VHAP's video preprocessing. VHAP's exported mesh-based masks and Metrical Tracker's crop-based alphas belong to the native pipeline recipes.

```bash
mkdir -p "$WORK/source"
ffmpeg -v error -i "$VIDEO" -vf fps=25 -start_number 0 "$WORK/source/%06d.png"
python scripts/prepare_common_canvas.py prepare \
  --rgb "$WORK/source" --alpha "$PREFIT_FULL_FRAME_ALPHA" \
  --splits "$SPLITS_JSON" --source "$SOURCE_VIDEO_ID" \
  --camera-id "$PHYSICAL_CAMERA_ID" --fps 25 --output "$WORK/common"
```

RGB and alpha filenames have numeric stems on the same zero-based decoded grid. Each input pair has identical full-frame dimensions. The split file contains `train`, `val`, and `test` lists of decoded frame IDs. Use disjoint time blocks and keep the train and test lists populated. The shared 512-pixel square uses centered padding and preserves aspect ratio. It contains the white-composited targets, alpha masks, uncomposited RGB for SMIRK, and a manifest with decoded times, hashes and the source-to-canvas affine. Keep these artifacts under an ignored run directory.

Write each tracker's verified mapping as a JSON object from native GA timestep (string) to decoded frame ID (integer). Verify it from source frame provenance and pixels. Metrical Tracker's GA exporter numbers checkpoint files consecutively, and `begin_frames` can remove initial source frames. VHAP exports original timestep indices, which should be checked against the decode and any subset. A match on filenames, timestep numbers, fps or frame counts alone is insufficient.

## 2. Adapt the tracker exports

Apply the common-canvas patch to the same GaussianAvatars checkout used with the project's usual dependency fixes. This is a Python patch and uses the existing CUDA rasterizer.

```bash
patch_file="$PWD/patches/ga-common-canvas.patch"
if ! git -C "$GA_ROOT" apply --reverse --check "$patch_file"; then
  git -C "$GA_ROOT" apply "$patch_file"
fi
```

Metrical Tracker uses the recorder's exclusive crop box. Pass it in `x0 y0 x1 y1` order. The tracker stores its native `bbox.pt` in `x0 x1 y0 y1` order. For a VHAP full-frame export the adapter infers the resize from the two raster dimensions. For both trackers, the adapter composes the native-to-canvas affine with the native intrinsics, keeps the exported camera pose, and links the existing FLAME files. The patched GA reader honors both focal lengths and the principal point in image-edge coordinates. The per-frame flag `camera_model: pinhole_image_edges` enables this camera model.

```bash
python scripts/prepare_common_canvas.py adapt \
  --common "$WORK/common" --native "$METRICAL_GA_EXPORT" --layout metrical \
  --metrical-bbox "$X0" "$Y0" "$X1" "$Y1" --frame-map "$METRICAL_FRAME_MAP" \
  --output "$WORK/mica"
python scripts/prepare_common_canvas.py adapt \
  --common "$WORK/common" --native "$VHAP_GA_EXPORT" --layout vhap \
  --frame-map "$VHAP_FRAME_MAP" --output "$WORK/vhap"
```

Run SMIRK on the common uncomposited images. This keeps its own landmark crop and removes the dependency on Metrical Tracker's crop, matting and camera. The export estimates its constant identity from the training subset. Its weak-perspective camera is approximated by a pinhole camera, using a fixed focal length of twice the canvas width by default and a nominal fixed-camera translation. Save the chosen focal in the protocol and check the weak-to-perspective residual on each sequence. The export leaves out SMIRK's eyelid offsets.

```bash
# Use the SMIRK environment here. REPO_ROOT and WORK should be absolute.
(cd "$SMIRK_ROOT" && PYTHONPATH=. python "$REPO_ROOT/scripts/static_rendering/track_folder_for_ga.py" \
  --image-dir "$WORK/common/tracking_images" --preserve-frame-shape \
  --output "$WORK/smirk_tracks.pt")

# Use the GaussianAvatars environment for export, diagnostics and rendering.
python scripts/export_smirk_to_ga.py --track-params "$WORK/smirk_tracks.pt" \
  --common-canvas "$WORK/common" --target "$WORK/smirk"
python scripts/diagnose_ga_mesh_overlay.py --dataset "$WORK/smirk" \
  --ga-root "$GA_ROOT" --track-params "$WORK/smirk_tracks.pt" \
  --frames 0,1,2 --out_dir "$WORK/smirk_overlay"
```

Use fresh output folders. The historical canvas commands remain available for a native-pipeline comparison.

## 3. Train and score the same inputs

Verify that all arms resolve to the same RGB and alpha files, same physical camera and decoded frame IDs, and same split assignment. Inspect full posed FLAME projections against the canvas. For Metrical Tracker and VHAP, compare the actual GA projection with the native pinhole projection transformed by the recorded affine. The rasterizer still stabilizes covariance using its symmetric offscreen clamp, so inspect clips with large principal point shifts for artifacts. For SMIRK, compare against the native weak projection and check upright orientation. These tests establish adapter behavior. An independent geometric reference is needed to evaluate tracking accuracy.

Use `--eval --bind_to_mesh --not_finetune_flame_params` across arms for controlled tracker comparisons. Fix effective resolution, initialization seed, regularizers, training and densification schedule, and training/evaluation frame membership. A common positive `--scene_extent` also fixes GA's scene-radius factor used by spatial learning rates and density control. Choose and record it before scoring in the shared FLAME metric scale. This controls the optimizer setting while preserving each tracker's estimated motion, mesh and projection.

Render the same held-out frames with the same display background. Compare the resulting `gt/` PNGs byte for byte across arms. Use the [manifest-based scorer](tracking_experiments.md#1-audit-and-group). Report the shared foreground or face/mouth metrics alongside frame metrics, since large uniform background regions contribute to full-frame scores. Use the same independently defined region for every tracker. Hold out complete sequences before making unseen-motion claims. A time block within a fitted sentence tests RGB frames withheld from GA training. Further restrict tracker fitting to the training frames if the comparison requires unseen frames for the tracker as well.

## Interpreting comparisons

- **Tracking-error sensitivity and recovery.** Use a fixed-avatar dose sweep for driving-time sensitivity. Cross training and driving conditions to measure recovery from a recurring tracking error. Report the identities, seeds, camera views and training budget.
- **Tracker comparison.** The shared canvas controls GA supervision, scoring pixels and timing. Each tracker retains its observation processing and camera or mesh model. Isolating geometry from estimated camera effects requires validated calibration and alignment.
- **Ranking transfer.** Pair independent geometry measurements and avatar metrics on the same performances. Report missing tracks, valid-frame coverage and identity-level variation.

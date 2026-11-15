# How tracking errors propagated to the rendered avatar

Questions: First, how does a known error in FLAME tracking change the images produced by a fixed GaussianAvatars model? Second, can training under that error help the avatar render correctly when the same error is present at evaluation? We ran these as separate experiments because a fixed model can show sensitivity to an error, while a matched training comparison can show whether optimization recovers some of the loss.

## Method

We traced the pipeline from the tracker through camera and FLAME export, GA training, rendering, and metric evaluation. Three findings shaped the experiments.

- The standard rendering commands had evaluated training images, and GA's default reader can merge validation and test views into training. We used explicit splits and `--eval`. We recorded camera and timestep in a manifest because the renderer's PNG names are output indices and the legacy scorer can choose a reference by filename fallback.
- Tracker setups differed in crops, masks, GA resolution and optimizer settings. MICA plus Metrical Tracker and SMIRK shared parts of the image pipeline, while VHAP used a native export. We chose matched within-avatar interventions for the causal experiment. Earlier tracker scores cover different datasets and cannot identify how tracking rank transfers to avatar quality.
- MultiREX's deployed evaluator measures expression deformation under a supplied neutral identity and region-wise rigid alignment. NoW measures static scan-to-mesh error. The archived NoW summary reversed mean and median, and VHAP covered 53 of 352 images. These datasets still provide separate geometry checks. The NeRSemble speech clips provide real RGB for the rendering study. We verified the existing camera/export fixes with five SMIRK projection tests and a MICA fixed-camera equivalence check on ten ns018 frames. The maximum sampled MICA camera-space difference was approximately 6.2 × 10⁻⁸ m.

We added [an audit and grouping CLI](../scripts/audit_tracking.py) that checks GA export paths, camera transforms, split integrity and exact render/reference pairing. It also imported 15,432 archived MultiREX observations across three trackers and eight identities. It accepts clip- and frame-level labels for appearance and motion and reports how many distinct identities, clips and observations support each group. Unannotated glasses, facial hair and motion labels remain unknown. The grouped mean weights frames within a clip, clips within an identity, and identities within a group. The historical stride-eight geometry outputs document the earlier protocol and have limited temporal bandwidth.

## Perturbations

We applied perturbations in [the shared FLAME state](../scripts/perturb_avatar.py) just before avatar deformation, after tracking and camera export. Each dose used the same zero-dose track, cameras, reference images and frame IDs. In fixed-avatar rendering we kept Gaussian weights and triangle bindings fixed. A FLAME root-pivot correction kept the rigid root transform fixed when a shape or expression change moved the root joint.

- **Identity shape and expression controls** add 0.25 or 0.75 to a selected FLAME coefficient. Coefficients describe model directions. We measured the resulting 3D displacement to give each dose a physical interpretation.
- **Jaw bias** composes a 1° or 3° rotation about the local jaw X axis. We measured effects around the mouth.
- **Local chin bias** adds a smooth downward rest-mesh displacement in a small chin region. The support excludes lips and mesh boundary vertices. Doses target 1 or 3 mm RMS displacement within that support.
- **Articulation lag** moves jaw and expression together relative to the original frame, while retaining head rotation, eye poses, camera and reference time. At the validated 25 fps export rate, 40 and 80 ms are one and two exported timesteps. Rotations use SLERP and expression coefficients use linear interpolation. We discard the first two timesteps from every dose in these lag comparisons and validated the exact shifts in the resulting arrays.

The zero-dose FLAME track was the geometric reference for intervention size. The recorded video was the reference for rendered image quality. For the fixed sweeps we defined the mouth and chin crops from the zero-dose mesh. For the retraining contrast we reapplied a single clean-drive chin crop to all four conditions, including those with biased drive geometry.

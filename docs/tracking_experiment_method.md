# How tracking errors propagated to the rendered avatar

A fixed-avatar dose sweep measures how an injected FLAME tracking error changes rendered images. A matched training comparison measures whether training with the error recovers fidelity when that error recurs at driving time.

## Method

- Use explicit splits and GA's `--eval` flag. Pair renders and references by camera and timestep in a manifest. GA render filenames enumerate outputs.
- Use [the shared canvas](common_canvas.md) to match scoring pixels, masks and splits across trackers. Match GA resolution and optimizer settings across arms. Run the mesh and camera diagnostics before training.
- MultiREX measures expression deformation under a supplied neutral identity and region-wise rigid alignment. NoW measures static scan-to-mesh error. Report these geometry measurements with their own dataset and protocol. A transfer claim requires independent geometry and rendered-avatar measurements on the same performances.

The [audit and grouping CLI](../scripts/audit_tracking.py) checks GA export paths, camera transforms, split integrity and render/reference pairing. It accepts clip- and frame-level labels and reports identity, clip and observation counts. The grouped mean weights frames equally within a clip, clips equally within an identity, and identities equally within a group.

## Perturbations

We applied perturbations in [the shared FLAME state](../scripts/perturb_avatar.py) just before avatar deformation, after tracking and camera export. Each dose used the same zero-dose track, cameras, reference images and frame IDs. In fixed-avatar rendering we kept Gaussian weights and triangle bindings fixed. A FLAME root-pivot correction kept the rigid root transform fixed when a shape or expression change moved the root joint.

- **Identity shape and expression controls** add 0.25 or 0.75 to a selected FLAME coefficient. Coefficients describe model directions. We measured the resulting 3D displacement to give each dose a physical interpretation.
- **Jaw bias** composes a 1° or 3° rotation about the local jaw X axis. We measured effects around the mouth.
- **Local chin bias** adds a smooth downward rest-mesh displacement in a small chin region. The support excludes lips and mesh boundary vertices. Doses target 1 or 3 mm RMS displacement within that support.
- **Articulation lag** moves jaw and expression together relative to the original frame, while retaining head rotation, eye poses, camera and reference time. At the validated 25 fps export rate, 40 and 80 ms are one and two exported timesteps. Rotations use SLERP and expression coefficients use linear interpolation. We discard the first two timesteps from every dose in these lag comparisons and validated the exact shifts in the resulting arrays.

The zero-dose FLAME track was the geometric reference for intervention size. The recorded video was the reference for rendered image quality. For the fixed sweeps we defined the mouth and chin crops from the zero-dose mesh. For the retraining contrast we reapplied a single clean-drive chin crop to all four conditions, including those with biased drive geometry.

# Bundled dependency fixes

The fixes listed in `manifest.json` are applied by
`bash scripts/setup_dependencies.sh`. See [SETUP.md](../SETUP.md).

`manifest.json` records the upstream URLs, exact base revisions and default
checkout locations. Clone the dependencies at those revisions before setup;
initialize GaussianAvatars submodules recursively. Custom checkout locations
are read from `config/paths.env`. Run setup before building CUDA extensions.
Upstream code retains its upstream license terms.

After dependency setup, apply the optional GaussianAvatars patches when the
workflow calls for them.

- [Common-canvas cameras](../docs/common_canvas.md#2-adapt-the-tracker-exports) use `ga-common-canvas.patch`.
- [Explicit training seeds](../docs/tracking_experiments.md#replicate-the-matched-study-across-seeds) use `ga-training-seed.patch`.

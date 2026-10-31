# Bundled dependency fixes

These files are used automatically by `bash scripts/setup_dependencies.sh`.
Users do not need to apply patches individually. See [SETUP.md](../SETUP.md).

`manifest.json` records the upstream URLs, exact base revisions and default
checkout locations. Clone the dependencies at those revisions before setup;
initialize GaussianAvatars submodules recursively. Custom checkout locations
are read from `config/paths.env`. Run setup before building CUDA extensions.
Upstream code retains its upstream license terms.

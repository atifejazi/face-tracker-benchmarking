# Dependency patches

`manifest.json` records upstream URLs, base commits and target directories.
Clone each repository at its recorded commit (initialize GA submodules
recursively). Apply once to clean checkouts; `--check` deliberately fails if a
patch is already applied or the checkout is incompatible. If following a newer
upstream revision, review conflicts rather than forcing a patch.

Load `config/paths.env`, then run the commands below. They use the default
ROOTDIR layout; substitute configured repository paths if yours differ.
Apply parent GA changes before the two submodule patches. Install/rebuild the
CUDA extensions afterwards. Model assets, datasets and generated actor configs
are not bundled. Upstream code retains its upstream license terms.

```bash
# gaussian-avatars: base 29b0021308d45c6dce0b522044a4893e189c7a23
git -C "$ROOTDIR/GaussianAvatars" apply --check "$REPO_ROOT/patches/gaussian-avatars.patch"
git -C "$ROOTDIR/GaussianAvatars" apply "$REPO_ROOT/patches/gaussian-avatars.patch"

# vhap: base 6e1e770e221d3c3d24498719308d21bd9e6e4c27
git -C "$ROOTDIR/VHAP" apply --check "$REPO_ROOT/patches/vhap.patch"
git -C "$ROOTDIR/VHAP" apply "$REPO_ROOT/patches/vhap.patch"

# mica: base af22e7a5810d474bc28a1433db533723d6bd2b07
git -C "$ROOTDIR/other/MICA" apply --check "$REPO_ROOT/patches/mica.patch"
git -C "$ROOTDIR/other/MICA" apply "$REPO_ROOT/patches/mica.patch"

# metrical-tracker: base c2a517244759d7eeaa0e84893d6748e75293f81a
git -C "$ROOTDIR/other/metrical-tracker" apply --check "$REPO_ROOT/patches/metrical-tracker.patch"
git -C "$ROOTDIR/other/metrical-tracker" apply "$REPO_ROOT/patches/metrical-tracker.patch"

# diff-gaussian-rasterization: base 59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d
git -C "$ROOTDIR/GaussianAvatars/submodules/diff-gaussian-rasterization" apply --check "$REPO_ROOT/patches/diff-gaussian-rasterization.patch"
git -C "$ROOTDIR/GaussianAvatars/submodules/diff-gaussian-rasterization" apply "$REPO_ROOT/patches/diff-gaussian-rasterization.patch"

# simple-knn: base 44f764299fa305faf6ec5ebd99939e0508331503
git -C "$ROOTDIR/GaussianAvatars/submodules/simple-knn" apply --check "$REPO_ROOT/patches/simple-knn.patch"
git -C "$ROOTDIR/GaussianAvatars/submodules/simple-knn" apply "$REPO_ROOT/patches/simple-knn.patch"
```

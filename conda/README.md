# Conda package and conda-forge submission

`recipe.yaml` is a conda recipe in the CEP-13/v1 format used by current
conda-forge infrastructure. It builds release 0.3.5 from the PyPI source
distribution, verifies that the archive has the expected SHA-256 digest,
installs the project with `pip`, and checks its imports, dependency metadata,
and command-line entry points without downloading the model weights.
The build removes the source distribution's generated `egg-info` directory so
setuptools regenerates its file list with the corrected package exclusions.

The 0.3.5 recipe carries a small patch that is also applied on this repository's
`main` branch. It keeps development/test namespaces out of the installed wheel,
declares the two directly imported plotting dependencies, and makes the existing
`megnet_qc_plots.py` entry point callable. Remove the patch after these changes
are included in a later tagged release.

## Build locally

Install `rattler-build`, then build against conda-forge:

```bash
conda create -n megnet-conda-build -c conda-forge rattler-build
conda activate megnet-conda-build
rattler-build build --recipe conda/recipe.yaml --channel conda-forge
```

The recipe intentionally does not run `megnet_init`: that command downloads
model weights from Hugging Face, while conda package tests must be repeatable
without access to an external service. Users run `megnet_init` after installing
the package, as they do for the PyPI distribution.

## Submit to conda-forge

New conda-forge packages are submitted through a pull request to
[`conda-forge/staged-recipes`](https://github.com/conda-forge/staged-recipes).
They are not pushed directly from this source repository.

1. Fork and clone `conda-forge/staged-recipes`.
2. Create a branch from its current `main` branch.
3. Create `recipes/megnet-neuro/` in that checkout.
4. Copy this repository's `conda/recipe.yaml` and
   `conda/fix-v0.3.5-packaging.patch` into `recipes/megnet-neuro/`.
5. Commit the recipe, push the branch to your fork, and open a pull request.
6. Address the conda-forge linter and reviewer feedback.

After the staged-recipes pull request merges, conda-forge automatically creates
the `megnet-neuro-feedstock` repository, renders its CI configuration, builds
the package, and publishes successful builds. Future conda packaging changes
belong in that feedstock rather than in staged-recipes.

For each later MEGnet-neuro release, update `version`, reset `build.number` to
zero, and replace `source.sha256` with the digest of the new tagged archive:

```bash
curl -L https://pypi.org/packages/source/m/megnet-neuro/megnet_neuro-VERSION.tar.gz \
  | sha256sum
```

## License note

The upstream `LICENSE` permits redistribution and use only for academic and
research purposes. The recipe therefore reports the project's existing custom
`LicenseRef-LICENSE` expression and packages the complete license text. Because
this is not a standard open-source license, conda-forge reviewers may ask the
copyright holder to clarify whether distribution through the public
conda-forge channel is permitted.

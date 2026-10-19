# Python Package Publishing Process

Releases are built and published by GitHub Actions (`.github/workflows/publish.yml`)
when a version tag is pushed. Publishing uses PyPI trusted publishing, so no API
tokens are needed.

## One-time setup

1. On PyPI, at https://pypi.org/manage/project/omnicloudmask/settings/publishing/,
   add a trusted publisher:
   - Owner: `DPIRD-DMA`
   - Repository name: `OmniCloudMask`
   - Workflow name: `publish.yml`
   - Environment: `pypi`
2. On GitHub, under Settings → Environments, create an environment named `pypi`.
   Optionally add required reviewers so each release needs approval.

## Making a release

1. Run the full test suite locally, including the tests CI skips. Some tests may be
   skipped depending on the available devices (CUDA, MPS).

   ```bash
   uv run pytest tests/
   ```

2. Move the `[Unreleased]` entries in `CHANGELOG.md` under the new version. There
   is no version file to bump: setuptools-scm takes the version from the git tag.
3. Merge to `main` and wait for CI to pass.
4. Tag the release on `main` and push the tag. The tag is `v` followed by the
   version; a suffix such as `v1.8.0rc1` publishes a prerelease.

   ```bash
   git tag v1.8.0
   git push origin v1.8.0
   ```

The workflow builds the sdist and wheel, publishes them to PyPI, and creates a
GitHub release with the files attached and generated release notes.

## conda-forge

The [conda-forge feedstock](https://github.com/conda-forge/omnicloudmask-feedstock)
builds from the PyPI sdist with `--no-build-isolation`, so it doesn't read the
build requirements from `pyproject.toml`. Its recipe must list them as host
requirements, currently `setuptools >=77` and `setuptools-scm >=8`. Without
setuptools-scm the build doesn't fail: the package installs as version 0.0.0,
which then fails `pip check` or a downstream `omnicloudmask>=...` requirement.

When a release changes the build or runtime requirements, update the recipe in the
bot's version PR, as the bot only changes the version and checksum.

## Manual fallback

To build and publish by hand, for example to test on Test PyPI:

```bash
rm -rf dist/
uv build
uv publish --publish-url https://test.pypi.org/legacy/
pip install -i https://test.pypi.org/simple/ omnicloudmask --extra-index-url https://pypi.org/simple
```

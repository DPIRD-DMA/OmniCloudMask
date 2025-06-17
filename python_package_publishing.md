# Python Package Publishing Process

## 1. Install Required Tools

Install the necessary tools for testing, building and uploading packages:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```
## 2. Run tests

Run the tests and make sure they all pass, depending on available devices some tests may be skipped.

```bash
pytest tests/
```
## 3. Remove old builds

Remove any old builds from the "dist" folder, so we upload only the newest release.

## 4. Build the Package

Build the OmniCloudMask package:

```bash
uv build
```

## 5. Upload to Test PyPI

Upload the package to Test PyPI for testing purposes. You will need to enter your user token:

```bash
uv publish --index-url https://test.pypi.org/legacy/
```

## 6. Test Installation

Install the package from Test PyPI in a new environment to ensure it works correctly:

```bash
pip install -i https://test.pypi.org/simple/ omnicloudmask --extra-index-url https://pypi.org/simple
```

## 7. Upload to PyPI

Once testing is complete and successful, upload the package to the main PyPI repository:

```bash
uv publish
```

> **Note:** Ensure you have the necessary credentials and permissions for both Test PyPI and PyPI before attempting to upload.
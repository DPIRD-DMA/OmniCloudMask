# Python Package Publishing Process

## 1. Install Required Tools

Install the necessary tools for building and uploading packages:

```bash
pip install build
pip install twine
```

## 2. Build the Package

Build the OmniCloudMask package:

```bash
python -m build
```

## 3. Upload to Test PyPI

Upload the package to Test PyPI for testing purposes. You will need to enter your user token:

```bash
twine upload --repository testpypi dist/*
```

## 4. Test Installation

Install the package from Test PyPI in a new environment to ensure it works correctly:

```bash
pip install -i https://test.pypi.org/simple/ omnicloudmask --extra-index-url https://pypi.org/simple
```

## 5. Upload to PyPI

Once testing is complete and successful, upload the package to the main PyPI repository:

```bash
twine upload --repository pypi dist/*
```

> **Note:** Ensure you have the necessary credentials and permissions for both Test PyPI and PyPI before attempting to upload.
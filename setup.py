from setuptools import find_packages, setup

setup(
    name="omnicloudmask",
    version="1.0.1",
    description="""Python library for cloud and cloud shadow segmentation in high to moderate resolution satellite imagery""",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Nick Wright",
    author_email="nicholas.wright@dpird.wa.gov.au",
    url="https://github.com/DPIRD-DMA/OmniCloudMask",
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=[
        "fastai>=2.7",
        "timm>=0.9,<1.0.4",
        "tqdm>=4.0",
        "rasterio>=1.3",
        "gdown>=5.1.0",
        "torch>=2.2",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_data={"omnicloudmask": ["models/model_download_links.csv"]},
)

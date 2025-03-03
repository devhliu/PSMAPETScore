from setuptools import setup, find_packages

setup(
    name="psmapetscore",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "psmapetscore.models": ["README.md"],
        "psmapetscore": ["models/**/README.md"]
    },
    include_package_data=True,
    install_requires=[
        "pydicom",
        "nibabel",
        "SimpleITK",
        "torch",
        "torchvision",
        "scikit-learn",
        "scikit-image",
        "pandas",
        "matplotlib",
        "pillow",
        "pylatex",
        "tqdm",
        "nnunetv2"  # Add nnUNetv2 dependency
    ],
    author="devhliu",
    author_email="devhliu@github.com",
    description="A Python package for automatic PSMA-PET structure reporting",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/devhliu/PSMAPETScore",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
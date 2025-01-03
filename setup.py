import os
from setuptools import setup, find_packages

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

version = {}
with open(os.path.join(here, "massspecgym", "__init__.py")) as f:
    exec(f.read(), version)

setup(
    name="massspecgym",
    packages=find_packages(),
    version=version['__version__'],
    description="MassSpecGym: Benchmark For the Discovery of New Molecules From Mass Spectra",
    author="MassSpecGym developers",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="",  # TODO: Add URL to documentation
    install_requires=[  # TODO: specify versions, requirements.txt
        "torch",
        "pytorch-lightning",
        "torchmetrics",
        "torch_geometric",
        "tokenizers",
        "numpy",
        "rdkit",
        "myopic-mces",
        "matchms",
        "wandb",
        "huggingface-hub",
        "seaborn",
        "standardizeUtils @ git+https://github.com/boecker-lab/standardizeUtils/#egg=standardizeUtils",
        "chemparse",
        "chemformula",
        "networkx"
    ],
    extras_require={
        "dev": [
            "black",
            "pytest",
            "pytest-cov",
        ],
        "notebooks": [
            "jupyter",
            "ipywidgets",
            "h5py",
            "scikit-learn",
            "pandarallel",
        ],
    }
)

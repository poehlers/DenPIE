from setuptools import setup, find_packages

# TODO: replace with the actual GitHub URL once the repo is created
HTTPS_GITHUB_URL = "https://github.com/<TODO-org>/<TODO-repo>"

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ["numpy", "scipy", "torch", "FrEIA", "pyyaml", "getdist",
                "nflows", "torchvision", "sbi", "optuna"]

setup(
    name="den_pie",
    version="1.0.1",
    author="Benedikt Schosser",
    author_email="schosser@stud.uni-heidelberg.de",
    description="Simulation based inference for 21cm cosmology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=HTTPS_GITHUB_URL,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
    entry_points={"console_scripts": ["den_pie=den_pie.__main__:main"]},
)

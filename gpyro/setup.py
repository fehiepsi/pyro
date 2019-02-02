from __future__ import absolute_import, division, print_function

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

EXTRAS_REQUIRE = [
    "jupyter",
    "matplotlib",
    "pandas",
    "seaborn",
    "torchvision",
]

setuptools.setup(
    name="gpyro",
    version="0.0.1",
    author="Pyro Development Team",
    author_email="pyro@uber.com",
    description="A Gaussian Process library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pyro-ppl/gpyro",
    packages=setuptools.find_packages(exclude=["tests*"]),
    install_requires=[
        "gpytorch",
        "pyro-ppl",
        "torch",
    ],
    extras_require={
        "extras": EXTRAS_REQUIRE,
        "test": [
            "pytest",
        ],
        "dev": EXTRAS_REQUIRE + [
            "flake8",
			"isort",
            "pytest",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

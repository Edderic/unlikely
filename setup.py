import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="unlikely",
    version="0.0.3",
    author="Edderic Ugaddan",
    author_email="edderic@gmail.com",
    description="Parallelized, Likelihood-free Bayesian Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/edderic/unlikely",
    project_urls={
        "Bug Tracker": "https://github.com/edderic/unlikely/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)

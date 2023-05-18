from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="rilacs",
    version="1.0.2",
    description="Risk limiting audits via confidence sequences",
    long_description=long_description,
    url="http://github.com/WannabeSmith/RiLACS",
    author="Ian Waudby-Smith",
    author_email="ianws@cmu.edu",
    license="BSD 3-Clause",
    packages=["rilacs"],
    install_requires=[
        "confseq",
        "numpy",
        "scipy",
    ],
    zip_safe=False,
)

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="rilacs",
    version="1.0.3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="Risk limiting audits via confidence sequences",
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

from setuptools import setup

setup(
    name="rilacs",
    version="0.1",
    description="Risk limiting audits via confidence sequences",
    url="http://github.com/WannabeSmith/RiLACS",
    author="Ian Waudby-Smith",
    author_email="ianws@cmu.edu",
    license="BSD 3-Clause",
    packages=["rilacs"],
    install_requires=["confseq @ https://github.com/WannabeSmith/confseq/tarball/master#egg=confseq", "numpy", "scipy"],
    zip_safe=False,
)

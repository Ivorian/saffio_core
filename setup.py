from setuptools import setup

setup(
    name="saffio_core",
    version="0.2.1",
    description="core",
    url="",
    author="Ivoria Nairov",
    author_email="",
    license="MIT",
    packages=["saffio_core"],
    install_requires=[
        "numpy",
        "pandas",
        "nltk",
        "pythainlp"
    ],
    zip_safe=False
)

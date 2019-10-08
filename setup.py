import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="push-ups",
    version="0.0.1",
    author="MIMUWRL",
    author_email="",
    description="RL toolkit for experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MIMUW-RL/push-ups",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

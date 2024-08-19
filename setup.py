import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="resoDesign", 
    version="0.1",
    author="Elena Acinapura",
    author_email="elena.acinapura@gmail.com",
    description="Python module for resonator design",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/HQClabo/dataAnalysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
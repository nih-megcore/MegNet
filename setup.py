import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MEGnet", 
    version="0.2",
    description="Create ICA features and classify ICA components",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nih-megcore/MegNET_2020",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: UNLICENSE",
        "Operating System :: Linux/Unix",
    ],
    install_requires=['openpyxl'],
    scripts=['MEGnet/prep_inputs/ICA.py'],
)

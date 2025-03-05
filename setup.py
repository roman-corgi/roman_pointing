import setuptools
import os.path
import re

with open("README.md", "r") as fh:
    long_description = fh.read()

with open(os.path.join("roman_pointing", "__init__.py"), "r") as f:
    version_file = f.read()

version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)

if version_match:
    version_string = version_match.group(1)
else:
    raise RuntimeError("Unable to find version string.")

setuptools.setup(
    name="roman_pointing",
    version=version_string,
    author="Dmitry Savransky",
    author_email="ds264@cornell.edu",
    description="Roman Observatory pointing calculation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roman-corgi/roman_pointing",
    packages=setuptools.find_packages(exclude=["tests*", "tools*", "Notebooks*"]),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)

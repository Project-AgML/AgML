import setuptools

with open("../README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="agml",
    version="0.0.1",
    author="UCD Plant AI and Biophysics Lab",
    author_email="jmearles@ucdavis.edu",
    description="Agricultural Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages()
)
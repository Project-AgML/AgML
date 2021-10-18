import os
import setuptools

# ============== SETUP INFORMATION ==============

with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r') as f:
    long_description = f.read()

python_supported = ['3', '3.7', '3.8', '3.9', '3.10']
os_supported = ['MacOS', 'Unix', 'POSIX', 'Microsoft :: Windows']

CLASSIFIERS = [
    "Development Status :: 3 - Alpha", # FIX: What is our actual development status?
    "Topic :: Scientific/Engineering",
    *[f"Operating System :: {os}" for os in os_supported],
    *[f"Programming Language :: Python :: {v}" for v in python_supported]
]

# ============ PACKAGE REQUIREMENTS =============

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
    install_requires = [line[:-1] for line in f.readlines() if line != '']

tests_require = [
    'pytest>=6.0.0',
    'pytest-order==1.0.0'
]

dev_requires = [
    'Shapely',
    'semantics',
    'scikit-image',
    'boto3',
    'botocore'
]

# ==================== SETUP ====================
    
setuptools.setup(
    name="agml",
    version="0.0.1",
    author="UC Davis Plant AI and Biophysics Lab", # FIX: We need an author/author email, or maintainer/maintainer_email
    maintainer_email="jmearles@ucdavis.edu",
    url="https://github.com/plant-ai-biophysics-lab/AgML", # FIX: Do we want to make a cover webpage for this?
    description="A comprehensive library for agricultural deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    extras_require={
        'dev': dev_requires,
        'test': tests_require
    }
)


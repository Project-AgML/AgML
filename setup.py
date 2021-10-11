import os
import setuptools

# ============== SETUP INFORMATION ==============

with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r') as f:
    long_description = f.read()

# ============ PACKAGE REQUIREMENTS =============

install_requires = [
    'pyyaml>=5.4.1'
]

tests_require = [
    'pytest>=6.0.0',
    'pytest-order==1.0.0'
]

backend_requires = [

]

# ==================== SETUP ====================
    
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


"""A setuptools based setup module for the if97 Python package"""
from setuptools import setup
from platform import python_version

# validate local python version supports requirements
major, minor = map(int, python_version().split('.')[:-1])
message = 'pyioflash requires python version 3.7 or greater'
assert major == 3 and minor >= 7, message

# get the long description from the README file
with open('readme.md') as readme:
    long_description = readme.read()

# define the setup configuration
setup(
    name = 'pyioflash',
    version = '1.0.34',
    description = 'A Python package for processing FLASH4 simulations.', 
    long_description = long_description,
    url = 'https://github.com/Balaras-Group/pyIOFlash',
    author = 'Aaron Lentner',
    author_email = 'aaronlentner@gwmail.gwu.edu',
    license = 'MIT',
    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Engineers',
        'Topic :: Scientific/Engineering :: Physics :: Fluid Dynamics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7 :: Only',
        ],
    keywords = 'FLASH HDF5 HPC Simulation CFD',
    packages=['pyioflash'],
    install_requirments = ['numpy', 'matplotlib', 'h5py'],
    entry_points = { },
    include_package_data = True,
    zip_safe = False
    )
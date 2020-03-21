"""A setuptools based setup module for the if97 Python package"""
from platform import python_version
from setuptools import setup

# validate local python version supports requirements
MAJOR, MINOR = map(int, python_version().split('.')[:-1])
MESSAGE = 'pyioflash requires python version 3.7 or greater'
assert MAJOR == 3 and MINOR >= 7, MESSAGE

# get the long description from the README file
with open('readme.rst') as readme:
    LONG_DESCRIPTION = readme.read()

# define the setup configuration
setup(
    name='pyioflash',
    version='1.0.41',
    description='A Python package for processing FLASH4 simulations.',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/Balaras-Group/pyIOFlash',
    author='Aaron Lentner',
    author_email='aaronlentner@gwmail.gwu.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Engineers',
        'Topic :: Scientific/Engineering :: Physics :: Fluid Dynamics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7 :: Only',
        ],
    keywords='FLASH HDF5 HPC Simulation CFD',
    packages=['pyioflash'],
    install_requirments=['numpy', 'matplotlib', 'h5py'],
    entry_points={},
    include_package_data=True,
    zip_safe=False
    )

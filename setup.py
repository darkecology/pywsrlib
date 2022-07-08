from setuptools import find_packages, setup

setup(
    name='wsrlib',
    version="0.1.0",
    description='(Biological) analysis of weather surveillance radar',
    packages=find_packages(include=['wsrlib']),
    url='https://github.com/darkecology/pywsrlib',
    author='Dan Sheldon',
    author_email='sheldon@cs.umass.edu',
    install_requires=[
        'arm_pyart==1.11.5',
        'boto3', # boto3-1.17.87
        'netCDF4==1.5.6',
        'scipy==1.5.4',
        'matplotlib==3.3.4',
        'pandas==1.1.5',
        'more-itertools==8.7.0',
    ],
    keywords='radar aeroecology ecology weather',
    license='MIT'
)

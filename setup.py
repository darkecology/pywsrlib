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
        'arm_pyart',
        'boto3'
    ],
    keywords='radar aeroecology ecology weather',
    license='MIT'
)

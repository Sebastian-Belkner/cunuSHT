import setuptools
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
import glob

with open("README.md", "r") as fh:
    long_description = fh.read()


def configuration(parent_package='', top_path=''):
    config = Configuration('', parent_package, top_path)
    return config

setup(
    name='cunusht',
    version='0.1',
    packages=['cunusht', 'cunusht.sht', 'cunusht.deflection'],
    url='https://github.com/Sebastian-Belkner/cunuSHT',
    author='Sebastian Belkner',
    author_email='to.sebastianbelkner@gmail.com',
    description='General spin-n SHTs on CPU and GPU',
    install_requires=[
        'numpy',
        'healpy',
        'logdecorator',
    ],
    requires=['numpy'],
    long_description=long_description,
    configuration=configuration)


from distutils.core import setup
from setuptools import find_packages

PACKAGE='hwolf'
NAME='hwolf'
VERSION='0.0.1'#__import__(PACKAGE).__version__
AUTHOR='clinjie'

setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    license='GNU Lesser General Public License v3.0',
    packages=find_packages(),
    zip_safe=False,
)

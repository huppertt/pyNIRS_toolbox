from setuptools import find_packages, setup

setup(
    name='pyBrainAnalyzIR',
    packages=find_packages(['pyBrainAnalyzIR']),
    version='0.1.0',
    description='Brain AnalyzIR Wrapper for Cedalion',
    author='T Huppert',
    install_requires=[],
    setup_requires=['pytest-runner'],
)
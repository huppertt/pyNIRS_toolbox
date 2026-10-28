from setuptools import find_packages, setup

setup(
    name='pyBrainAnalyzIR',
    packages=find_packages(),
    version='0.1.0',
    description='Brain AnalyzIR Wrapper for Cedalion',
    author='T Huppert',
    package_data={
        'pyBrainAnalyzIR.media': ['data/*.npz'],
        'pyBrainAnalyzIR.testing': ['matlab/*.m'],
        'pyBrainAnalyzIR.vis': ['data/*.elc', 'data/*.txt'],
    },
    include_package_data=True,
    install_requires=[],
    setup_requires=['pytest-runner'],
)

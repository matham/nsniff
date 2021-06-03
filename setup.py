from setuptools import setup, find_packages
import nsniff

with open('README.rst') as fh:
    long_description = fh.read()

setup(
    name='nsniff',
    version=nsniff.__version__,
    author='Matthew Einhorn',
    author_email='matt@einhorn.dev',
    url='https://matham.github.io/nsniff/',
    license='MIT',
    description='Records odor sense data.',
    long_description=long_description,
    classifiers=['License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering',
                 'Topic :: System :: Hardware',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 ],
    packages=find_packages(),
    install_requires=[
        'base_kivy_app', 'kivy', 'nixio==1.4.9', 'pyserial',
        'kivy_garden.graph', 'trio', 'pymoa-remote', 'pymoa',
        'tree-config', 'kivy_trio'],
    extras_require={
        'dev': [
            'pytest>=3.6', 'pytest-cov', 'flake8', 'sphinx-rtd-theme',
            'coveralls', 'pytest-trio',
            'pytest-kivy', 'pytest-dependency', 'docutils',
            'sphinx'],
    },
    package_data={'nsniff': ['data/*', '*.kv']},
    entry_points={'console_scripts': ['nsniff=nsniff.main:run_app']},
)

#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='imprl',
    author='Prateek Bhustali',
    description='Inspection and Maintenance Planning with Reinforcement Learning',
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={'': ['*.yaml']},  # All packages should include any .yaml files
    classifiers=[
    'Programming Language :: Python :: 3',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Mathematics',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    ],
)

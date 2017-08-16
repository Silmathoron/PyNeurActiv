#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import sys
from setuptools import setup, find_packages


setup(
        name = 'PyNeurActiv',
        version = '0.1.0',
        description = 'Python tools and theoretical models for neuronal activity analysis',
        package_dir = {'': '.'},
        packages = find_packages('.'),

        # Requirements
        install_requires = ['numpy', 'scipy>=0.11', 'matplotlib'],
        extras_require = {
            'pandas': 'pandas'
        },

        # Metadata
        url = 'https://github.com/SENeC-Initiative/PyNeurActiv',
        author = 'Tanguy Fardet',
        author_email = 'tanguy.fardet@univ-paris-diderot.fr',
        license = 'GPL3',
        keywords = 'neural activity analysis simulation'
)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import pii_neural_network

setup(
    name = 'pii-neural-network',
    version = '0.0.1',
    packages = find_packages(),
    author = 'Émilie ROGER',
    author_email = 'emilie.roger@ensc.fr',
    description = 'Permet de créer des réseaux de neurones',
    long_description = open('README.md').read(),
    include_package_data = True,
    url = 'https://github.com/remilieam/pii-neural-network',
    classifiers = [
        "Programming Language :: Python",
        "License :: OSI Approved",
        "Natural Language :: French",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.5",
    ],
    license = "WTFPL",
)

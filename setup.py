#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

requirements = [
    'sparse',
    'tensorly',
    'annoy',
    'pandas',
    'tqdm',
    'flask',
    'spacy',
    'hottbox',
    'numpy',
    'tabulate',
    'torch'
]

setup(
    name='sampo',
    version='0.0.1',
    description=('Sampo is a tool for automatically building ' +
                 'knowledge bases from review corpora.'),
    author="Megagon Labs",
    packages=[
        'sampo'
    ],
    package_dir={'sampo': 'sampo'},
    include_package_data=True,
    install_requires=requirements,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='knowledge-base construction, review understanding',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)

#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/19/21

import imblearn
from pathlib import Path

from setuptools import setup, find_namespace_packages

long_description = Path('README.adoc').read_text(encoding='utf-8', errors='ignore')

classifiers = [  # copied from https://pypi.org/classifiers/
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3 :: Only',
]
keywords = [
    'machine learning', 'imbalanced learning', 'classification', 'rare phenomenon learning',
    'NLP', 'natural language processing,', 'computer vision'
]
packages = find_namespace_packages()
setup(
    name='imblearn',
    version=imblearn.__version__,
    description=imblearn.__description__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=classifiers,
    python_requires='>=3.7',
    url='https://github.com/thammegowda/011-imb-learn',
    download_url='https://github.com/thammegowda/011-imb-learn',
    platforms=['any'],
    author='Thamme Gowda',
    author_email='tgowdan@gmail.com',
    packages=packages,
    keywords=keywords,
    entry_points={
        'console_scripts': [
            'imblearn-pipe=imblearn.pipeline:main'
        ],
    },
    install_requires=[
        'torch==1.9.0',
        'ruamel.yaml==0.17.4',
        'torchvision==0.9.1',
        'transformers==4.5.1',
        'datasets==1.6.2',
        'tensorboard==2.5.0',
        'nlcodec==0.4.0',
    ],
    # include_package_data=True,
    # zip_safe=False,
)

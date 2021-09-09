#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
from setuptools import setup, find_packages


requirements = ["torch>=1.1.0", "boto3","pytorch_pretrained_bert==0.6.1","requests","tqdm","seqeval>=0.0.5","numpy"]

if sys.version_info[:2] < (2, 7):
    requirements.append('argparse')
if sys.version_info[:2] < (3, 4):
    requirements.append('enum34')
if sys.version_info[:2] < (3, 5):
    requirements.append('typing')

extras_require = {
    ':python_version<"2.7"': ['argparse'],
    ':python_version<"3.4"': ['enum34'],
    ':python_version<"3.5"': ['typing'],
}

setup(
    name="sikufenci",
    version="0.0.28",
    author="Liu Chang",
    author_email="649164915@qq.com",
    description="NLP tool for Ancient Chinese word segmentation.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    license="MIT",
    url="https://github.com/SIKU-BERT/sikufenci",
    keywords=['classical-chinese', 'wordseg', 'nlp'],
    packages=find_packages(),
    install_requires=requirements,
    extras_require=extras_require,
    python_requires='>=2.6, >=3',
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Utilities',
        'Topic :: Text Processing',
    ]
)
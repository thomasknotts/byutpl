"""
Setup script for the lpfgopt package
"""

from setuptools import setup, find_packages

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
VERSION = '1.0.0'


def readme():
    with open(ROOT+'/README.md', encoding='utf8') as f:
        return f.read()

config = {
    'name'                 : 'byu_tpl',
    'description'          : 'Brigham Young University Thermophysical Properties Laboratory',
    'version'              : VERSION,
    'author'               : 'Thomas Allen Knotts IV',
    'author_email'         : 'thomas.knotts@byu.edu',
    'url'                  : 'http://knotts.byu.edu/',
    'download_url'         : 'https://github.com/thomasknotts/byutpl/releases',
    'python_requires'      : '>=3.6',
    'install_requires'     : ['numpy', 'scipy'],
    'packages'             : find_packages(),
    'scripts'              : [],
    'long_description'     : readme(),
    'long_description_'\
        'content_type'     : 'text/markdown',
    'include_package_data' : True,
    'license'              : 'GPLv3',
    "keywords"             : 'thermodynamics properties engineering',
    'classifiers'          : [
                                'Development Status :: 5 - Production/Stable',
                                'Intended Audience :: Science/Research',
                                'Intended Audience :: Education',
                                'Intended Audience :: End Users/Desktop',
                                'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                             ],
}

setup(**config)

from setuptools import setup, find_packages


NAME = 'energymodel'
AUTHOR = 'shuiruge'
AUTHOR_EMAIL = 'shuiruge@whu.edu.cn'
URL = 'https://github.com/shuiruge/energymodel'
VERSION = '0.1.0'


setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    packages=find_packages(exclude=[
        'tests.*', 'tests',
        'examples.*', 'examples',
        'dat.*', 'dat']),
    classifiers=[
        'Programming Language :: Python :: 3+',
    ],
    zip_safe=False,
)

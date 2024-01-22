from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as fh:
    install_requires = fh.read().strip().split('\n')

setup(
    name='fespy',
    version='0.1.0',
    packages=find_packages(),
    install_requires=install_requires,
    long_description=long_description,
    author='Mohammad Abdulqader',
    author_email='m.abdulqader172@gmail.com',
    description='Analysis module for computing thermodynamic and structural properties of proteins.',
    url='https://github.com/mabdulqader172/FESpy',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)

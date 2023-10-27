from setuptools import setup, find_packages

setup(
    name='recolect',
    version='0.0.1',
    packages=find_packages(
        exclude=['tests', 'tests.*', 'examples', 'examples.*']
    ),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn'
    ],
    entry_points={
        'console_scripts': [
            'recolect = recolect.__main__:main'
        ]
    },
    author='Uka Osim',
    description='A toolbox for building recommender systems',
    url='https://github.com/osimuka/recolect',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

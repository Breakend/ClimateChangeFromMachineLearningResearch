import sys
import subprocess
from setuptools import setup, find_packages
from distutils.version import LooseVersion


from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('climate_impact_tracker/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

setup(name='climate_impact_tracker',
      packages= find_packages(),
      install_requires=[
          'bs4',
          'shapely',
          'commentjson',
          'scipy',
          'joblib',
          'numpy',
          'pandas',
          'matplotlib',
          'py-cpuinfo',
          'pylatex'
      ], 
      extras_require={
        'tests': [
            'pytest==3.5.1',
            'pytest-cov',
            'pytest-env',
            'pytest-xdist',
        ],
        'docs': [
            'sphinx',
            'sphinx-autobuild',
            'sphinx-rtd-theme'
        ]
      },
      description='A toolkit for easy evaluation of Deep RL algorithms.',
      author='Peter Henderson',
      url='https://github.com/Breakend/DeepRLEvaluationToolkit',
      author_email='peter.henderson@mail.mcgill.ca',
      keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
               "gym openai baselines toolbox python data-science",
      license="MIT",
      version=main_ns['__version__'],
      scripts=['scripts/compute-tracker', 'scripts/track-impact']
      )

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*

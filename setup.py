import sys
import subprocess
from setuptools import setup, find_packages
from distutils.version import LooseVersion


if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

# Check tensorflow installation to avoid
# breaking pre-installed tf gpu
install_tf, tf_gpu = False, False
try:
    import tensorflow as tf
    if tf.__version__ < LooseVersion('1.5.0'):
        install_tf = True
        # check if a gpu version is needed
        tf_gpu = tf.test.is_gpu_available()
except ImportError:
    install_tf = True
    # Check if a nvidia gpu is present
    for command in ['nvidia-smi', '/usr/bin/nvidia-smi', 'nvidia-smi.exe']:
        try:
            if subprocess.call([command]) == 0:
                tf_gpu = True
                break
        except IOError:  # command does not exist / is not executable
            pass

tf_dependency = []
if install_tf:
    tf_dependency = ['tensorflow-gpu>=1.5.0'] if tf_gpu else ['tensorflow>=1.5.0']
    if tf_gpu:
        print("A GPU was detected, tensorflow-gpu will be installed")

setup(name='climate_impact_tracker',
      packages= find_packages(),
      install_requires=[
          'scipy',
          'joblib',
          'numpy',
          'pandas',
          'matplotlib',
          'py-cpuinfo'
      ] + tf_dependency,
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
      version="0.1"
      )

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*

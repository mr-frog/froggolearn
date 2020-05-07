from setuptools import setup, find_packages


setup(name='froggolearn',
      version='0.1',
      description='badly coded machine learning library',
      install_requires=[
        'pandas>=1.0.3',
        'numpy>=1.18.2',
        'libsvm>=3.23.0.4',
        'scipy>=1.4.1'
        ],
      url='',
      author='Mr Frog',
      author_email='',
      license='',
      packages=find_packages(),
      zip_safe=False)

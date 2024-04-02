from setuptools import setup, find_packages

setup(name='cddd',
      version='0.1',
      packages=find_packages(include=['cddd', 'cddd.*']),
      zip_safe=False)

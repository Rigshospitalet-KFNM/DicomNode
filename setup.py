from setuptools import setup, find_packages

print(find_packages(where="src"))

setup(name='dicomnode',
  version='0.0.1',
  description='Test',
  author='Christoffer Vilstrup Jensen',
  author_email='christoffer.vilstrup.jensen@regionh.dk',
  package_dir={"":"src"},
  packages=find_packages(where="src", exclude=["bin", "tests"]),
  install_requires=[
    'numpy>=1.23.0',
    'pydicom>=2.3.0',
    'pynetdicom>=2.0.2'
  ],
  extras_require = {
    "test" : ["coverage"]
  },

  entry_points={
  'console_scripts': [
   'omnitool=dicomnode.bin.omnitool:entry_func'
  ]},
)
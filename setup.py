from setuptools import setup, find_packages

print(find_packages(where="src"))


if __name__ == '__main__':
  setup(name='dicomnode',
    version='0.0.2',
    description='Test',
    author='Christoffer Vilstrup Jensen',
    author_email='christoffer.vilstrup.jensen@regionh.dk',
    package_dir={"":"src"},
    packages=find_packages(where="src", exclude=["bin", "tests"]),
    install_requires=[
      'pydicom>=2.3.1',
      'pynetdicom>=2.0.2',
      'psutil>=5.9.2',
      'typing_extensions>=4.4.0',
      'pylatex[matrices, matplotlib]'
    ],
    extras_require = {
     "test" : ["coverage", "coverage-lcov"],

    },
    python_requires='>=3.9.1',
    entry_points={
      'console_scripts': [
      'omnitool=dicomnode.bin.omnitool:entry_func'
    ]},
  )


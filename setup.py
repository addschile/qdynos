from setuptools import find_packages, setup

with open('qdynos/version.py') as f:
  exec(f.read(), globals())

description = ('Small package for performing quantum dynamics on small to medium systems.')

# Reading long Description from README.md file.
with open("README.md", "r") as fh:
  long_description = fh.read()

# Read in requirements
requirements = [
    requirement.strip() for requirement in open('requirements.txt').readlines()
]

setup(
    name='qdynos',
    version=__version__,
    url='http://github.com/addschile/qdynos',
    author='Addison J. Schile',
    author_email='addschilel@gmail.com',
    python_requires=('>=3.6.0'),
    install_requires=requirements,
    license='MIT',
    description=description,
    packages=find_packages(),
)

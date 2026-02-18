from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements if req.strip() and req.strip() != HYPEN_E_DOT]
    return requirements
setup(
    name='ml-project',
    version='0.1',
    author='Lokesh',
    description='A machine learning project',
    author_email='mail2lokesh022@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)
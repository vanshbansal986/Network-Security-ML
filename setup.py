from setuptools import find_packages , setup
from typing import List

def get_requirements() -> List[str]:
    """
    This function will return list of requirements
    """

    req_list = []

    try:
        with open("requirements.txt" , 'r') as file:
            # read each line from the file
            lines = file.readlines()

            # process the lines
            for line in lines:
                requirement = line.strip()

                # ignore empty lines and -e.
                if requirement and requirement != '-e .':
                    req_list.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")
    
    return req_list

setup(
    name = "NetworkSecurity",
    version = "0.0.1",
    author = "Vansh Bansal",
    author_email = "vanshabansal986@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements()
)


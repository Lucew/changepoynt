from setuptools import setup
from setuptools import find_packages
from os import path


# read the contents of requirements.txt
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()
with open(path.join(this_directory, 'requirements_tests.txt'), encoding='utf-8') as f:
    requirements_tests = f.read().splitlines()

# read content of the readme
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    print(long_description)


def main():

    setup(
        name="changepoynt",
        version="0.1.1",
        author="Lucas Weber",
        author_email="weber-lucas@web.de",
        url="https://github.com/Lucew/changepoynt",
        description="Several change point detection methods implemented in python.",
        long_description=long_description,
        long_description_content_type='text/markdown',
        zip_safe=False,
        include_package_data=True,
        packages=find_packages(exclude=['tests']),
        install_requires=requirements,
        dependency_links=[],
        tests_require=requirements_tests,
        extras_require={'test': requirements_tests},
        setup_requires=[],
        license="BSD License",
        test_suite="tests",
        classifiers=[
            "Programming Language :: Python :: 3.9",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering",
        ],
        keywords='changepoint timeseries engineering'
    )


if __name__ == '__main__':
    main()

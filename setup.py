from setuptools import setup
from setuptools import find_packages


def main():

    setup(
        name="changepoynt",
        version="0.0.1",
        author="Lucas Weber",
        author_email="",
        url="",
        description="Readable package for several change point detection methods implemented in python.",
        long_description="This package contains several readable change point detections methods in python.",
        zip_safe=False,
        include_package_data=True,
        packages=find_packages(),
        install_requires=[
            "numpy",
            "scipy",
            "numba",
        ],
        dependency_links=[],
        tests_require=[],
        setup_requires=[],
        license="MIT",
        test_suite="tests",
        classifiers=[
            "Programming Language :: Python :: 3.9",
            "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
            "Topic :: Scientific/Engineering",
        ]
    )


if __name__ == '__main__':
    main()

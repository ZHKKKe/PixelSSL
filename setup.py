import os
import setuptools

version_file = './pixelssl/version.py'
requirements_file = './pixelssl/requirements.txt'


def read_version_file():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()


def read_requirements_file():
    with open(requirements_file, 'r') as f:
        requirements = f.readlines()
    requirements = [r.strip() for r in requirements]
    return requirements


if __name__ == '__main__':
    info = read_version_file()
    requirements = read_requirements_file()

    setuptools.setup(
        name=info['__name__'],
        version=info['__version__'],
        description=info['__description__'],
        url=info['__url__'],
        license=info['__license__'],
        author=info['__author__'],
        author_email=info['__author_email__'],
        python_requires='>=3, <4',
        install_requires=requirements,
        packages=setuptools.find_packages(),
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 3',
        ],
    )

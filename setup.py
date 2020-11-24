from setuptools import setup, find_packages

setup(
    name = 'AutoRT',
    version = '2.0.0-beta',
    keywords='Retention time',
    description = 'Retention time prediction using deep learning',
    license = 'GNU General Public License v3.0',
    url = 'https://github.com/bzhanglab/autort',
    author = 'Bo Wen',
    author_email = 'bo.wen@bcm.edu',
    packages = find_packages(),
    include_package_data = True,
    platforms = 'any',
    install_requires = [
        'tensorflow'
    ],
)

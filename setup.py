from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='clustertools',
    packages = find_packages(),
    include_package_data=True,
    version='0.1.0',
    description='',
    long_description=readme(),
    classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3'],
    url='https://github.com/clustertoolsgroup/clustertools_project',
    author='',
    author_email='',
    install_requires=['numpy', 'matplotlib', 'scipy'],
    tests_require=['nose'],
    test_suite='nose.collector'
)

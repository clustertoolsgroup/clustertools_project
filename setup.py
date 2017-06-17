from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='clustertools',
    version='0.1.0',
    description='',
    long_description=readme(),
    classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3'],
    url='https://github.com/ TODO',
    author='',
    author_email='',
    packages=['clustertools'],
    install_requires=['numpy', 'matplotlib', 'scipy', 'pandas'],
    tests_require=['nose'],
    test_suite='nose.collector'
)

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "authorship-pkg-ernesto-martinez-del-pino",
    version = "0.0.1",
    author = "Ernesto Martínez del Pino",
    author_email = "ernestomar1997@hotmail.com",
    long_description = long_description,
    url = 'https://github.com/rojo1997/Authorship',
    packages = setuptools.find_packages(),
    clasifiers = [
        "Programming Language :: Python :: 3",
    ],
    python_requires = '>3.6'
)
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "Authorship",
    version = "0.1",
    author = "Ernesto Martínez del Pino",
    author_email = "ernestomar1997@correo.ugr.es",
    long_description = long_description,
    url = 'https://github.com/rojo1997/Authorship',
    packages = ['Authorship'],
    install_requires = [i.strip() for i in open('requirements.txt').readlines()],
    test_suite = 'tests',
    clasifiers = [
        "Programming Language :: Python :: 3",
    ],
    python_requires = '>3.6'
)
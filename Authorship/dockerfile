FROM python:latest as authorship_python

ADD . /authorship

RUN pip3 install -r /authorship/requirements.txt && mkdir /authorship/data /authorship/data/iniciativas08

RUN python -c 'import nltk; nltk.download("stopwords")'

RUN apt-get update && apt-get install curl

RUN apt install -y /authorship/exe/megacmd-Debian_10.0_amd64.deb

RUN mega-get https://mega.nz/folder/6MtVRK6Z#puzBanuLkwCxb0wCOXMaQg /authorship/
#RUN mega-get https://mega.nz/folder/CYkn1YaJ#WVoMZQU3Xe1TELU4OsYGpQ /authorship/data/

ENTRYPOINT [ "bash" ]
#ENTRYPOINT [ "python", "/authorship/src/main.py"]
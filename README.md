**Estado**: En desarrollo

# Autoría de documentos

Trabajo final de Grado para la identificación de la autoría de documentos. Haciendo uso de **Text Mining** y modelos de **Machine Learning**.

## Uso

Estracción e identificación de características sobre la escritura de un sujeto.

## Requerimientos

1. Python 3.6
2. [sklearn](https://github.com/scikit-learn/scikit-learn)
2. [nlkt](https://github.com/nltk/nltk)
3. [pandas](https://github.com/pandas-dev/pandas)

## Instalación

Instalación de sklearn
```
python -m pip install sklearn
```

Instalación de nltk
```
python -m pip install nltk
```
Instalación de pandas
```
python -m pip install pandas
```

## Conjunto de datos

Se ha testeado el modelo sobre el conjunto de datos de iniciatiavas del congreso 2008.

[Dataset](http://www.senado.es/web/actividadparlamentaria/iniciativas/detalleiniciativa/documentos/index.html;jsessionid=fKQKp9vDxNknrvpmnMTcFSb8QhDqRvZ156xPByyQ80qGcyGpRJGX!981478430?legis=8&id1=621&id2=000136)

El conjunto de datos se encuentra nativamente en formato XML. Tras eliminar los datos superfluos del mismo, la etiqueta y los documentos se han pasado a formato CSV. Siguiendo esta estructura:

| Etiqueta  | Texto                   |
| --------- |:-----------------------:|
| Persona 1 | Parrafos concatenados 1 |
| Persona 2 | Parrafos concatenados 2 |
| Persona 1 | Parrafos concatenados 3 |

## Ficheros

Fichero principal que realiza la lectura de los datos, la división train test y las llamadas fit y predict necesarias.

[main.py](../src/main.py)

Fichero que contiene el modelo. Ajuste de hiperparámetros, cross-validation, lematización, analizador de frases, TF-IDF y Linear Support Vector Machine. Todo se encuentra recogido en un **pipeline** único.

[Authorship.py](../src/Authorship.py)

Fichero para imprimir información deseada dentro de un pipeline.

[NVarPrint.py](../src/Authorship.py)


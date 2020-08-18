**Estado**: En desarrollo

# Autoría de documentos

Trabajo Fin de Grado sobre el estudio de la Identificación de la Autoría de Documentos (Authorship). Haciendo uso de **Text Mining**, modelos de **Machine Learning** y **Deep Learning**. El problema Authorship es un problema de **Aprendizaje Automático** de clasificación multietiqueta, donde la característica más importante o única se presenta en forma de texto libre.

# Documentación

La documentación extensa del proyecto se encuentra en el [trabajo](Authorship/doc/TFG%20Estudio%20de%20identificacion%20de%20autoría.pdf) presentado a la **Universidad de Granada** sobre la cual se realizo la defensa obteniendo una **calificación de 9,8**.

## Uso

El paquete provee de una extensión de la librería **sklearn** dedicada al **Procesamiento del Lenguaje Natural**.

## Requerimientos

Versión del lenguaje de programación:
* Python 3.6

Librerías de python:
1. [sklearn](https://github.com/scikit-learn/scikit-learn)
2. [nlkt](https://github.com/nltk/nltk)
3. [pandas](https://github.com/pandas-dev/pandas)
4. [dill](https://pypi.org/project/dill/)
5. [tensorflow](https://www.tensorflow.org/?hl=es-419)
6. [multiplocess](https://pypi.org/project/multiprocess/)
7. [googletrans](https://pypi.org/project/googletrans/)


## Instalación

La instalación se puede realizar mediante el fichero [requirements.txt](requirements.txt):

```
python -m pip -r requirements.txt
```

## Conjunto de datos

Se han testeado los modelos sobre el conjunto de datos de iniciatiavas del congreso 2008.

[Dataset](http://www.senado.es/web/actividadparlamentaria/iniciativas/detalleiniciativa/documentos/index.html;jsessionid=fKQKp9vDxNknrvpmnMTcFSb8QhDqRvZ156xPByyQ80qGcyGpRJGX!981478430?legis=8&id1=621&id2=000136)

El conjunto de datos se encuentra nativamente en formato XML. Tras eliminar los datos superfluos del mismo, la etiqueta y los documentos se han pasado a formato CSV. Siguiendo esta estructura:

| Etiqueta  | Texto                   |
| --------- |:-----------------------:|
| Persona 1 | Parrafos concatenados 1 |
| Persona 2 | Parrafos concatenados 2 |
| Persona 1 | Parrafos concatenados 3 |

## Organización del proyecto

El paquete **Authorship** engloba el conjunto de herramientas relacionadas con NLP que se han aplicado expresadas por el siguiente esquema:
1. [Preprocesamiento](Authorship/preprocessing.py)
2. [Selección de características](Authorship/feature_selection.py)
3. [Extracción de características](Authorship/feature_extraction/text.py)
4. [Redes neuronales](Authorship/neural_network.py)

Un fichero de [funciones](Authorship/functions.py) auxiliares para la lectura de datos y limpieza del dataset.

Puesto que la motivación del proyecto es estudiar la influencia de las distintas herramientas o transformaciones en tareas de clasificación los modelos finales se encuentra implementados en la sección de testing:
* [Authorship_test](tests/Authorship_test.py)

## Referencias de metodología

La metodología aplicada a lo largo del proyecto para la presentación de resultados ha seguido los estándares de división entre entrenamiento y test así como la aplicación de la técnica de validación cruzada.

Por otro lado, la guía que se ha seguido ha sido la aportada por Google.
* [Guía Google](https://developers.google.com/machine-learning/guides/text-classification?hl=es-419)


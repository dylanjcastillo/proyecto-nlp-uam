# Proyecto Clasificación de tweets para Univ. Autónoma Madrid

## 1. Introducción

El objetivo principal de este proyecto es clasificar los tweets publicados por los ayuntamientos españoles en base a una taxonomías determinada, utilizando modelos de Aprendizaje Automático (Machine Learning), para el análisis posterior de dichos tweets.

Para ello se obtuvieron los últimos 3.200 tweets publicados por las cuentas oficiales de los distintitos ayuntamientos españoles a través de la API de twitter, una vez obtenidos los datos se procedió al etiquetado de una muestra representativa de estos para poder entrenar modelos que permita etiquetar el resto de los tweets.

## 2. Extracción y etiquetado de datos

Se extrajeron los últimos 3.200 tweets publicados por las cuentas oficiales de los ayuntamientos españoles a través de la API de Twitter. Una vez obtenidos se procedió al etiquetado de una muestra representativa de los mismos utilizando [Doccano](https://github.com/doccano/doccano), para luego crear los modelos de clasificación con dichos datos.

En total, se extrajeron 467.073 tweets, se tradujeron (aprox. 20% no estaban en castellano) y se creo una [muestra estratificada](https://es.wikipedia.org/wiki/Muestreo_estratificado) (a nivel de ayuntamientos) de 2.264 tweets, que posteriormente fueron etiquetados manualmente.

Los tweets etiquetados se encuentran disponibles en el fichero `all_multiclass_20220911.json`.

### 2.1 Taxonomía utilizada

Para clasificar los tweets, se definió una taxonomía de tres niveles y se creó un modelo de aprendizaje automatizado de clasificación para cada una de las capas.

Esta es la estructura definida:

- **Capa 1**
  - Transparencia
  - Colaboración Ciudadana
  - Fomento de la participación
- **Capa 2**
  - Decisiones
  - Agradecimientos y Actos Simbólicos
  - Estado de Servicio
- **Capa 3**
  - Resultados
  - Contenido
  - Racionalidad

## 3. Creación de modelos predictivos

Se creó un modelo de clasificación por cada capa definida en la sección anterior. Para ello, se utilizaron las librerías de Python [scikit-learn](https://scikit-learn.org/stable/) y [transformers](https://huggingface.co/docs/transformers/index), que contienen una gran variedad de algoritmos de aprendizaje automático especializados en procesamiento de lenguaje natural. En concreto, se han utilizado los siguientes algoritmos:

1. **Capa 1**: Modelo [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) haciendo un ajuste fino con los datos disponibles.
2. **Capa 2**: Modelo [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) haciendo un ajuste fino con los datos disponibles.
3. **Capa 3**: Modelo de regresión logística, combinado con [embeddings](https://huggingface.co/hiiamsid/sentence_similarity_spanish_es) creados a partir de BETO, pero entrenados para evaluar similaridad de texto.

Adicionalmente, se hizó análisis de sentimiento de los tweets utilizando la librería de Python [pysentimiento](https://huggingface.co/finiteautomata/beto-sentiment-analysis).

### 3.1 Preprocesamiento

Se ha utilizado el mismo procedimiento: traducir los tweets que no están en castellano a castellano, quitar enlaces, reemplazar emojis por palabras y quitar "@".

### 3.2 Entrenamiento de modelos

Para las **capas 1 y 2**:

1. Se siguió un proceso de validación cruzada de 10 folds. En cada fold, se entrenó un modelo durante 6 epochs, con una tasa de aprendizaje de 4 x 10e-5, y se eligió el mejor modelo utilizando la métrica [F1](https://en.wikipedia.org/wiki/F-score).
2. Una vez entrenados los modelos por fold, se generan las predicciones de cada modelo y se elige la predicción definitiva utilizando un proceso de [votación suave](https://machinelearningmastery.com/voting-ensembles-with-python/).

Para la **capa 3**:

1. Se siguió un proceso de validación cruzada de 10 folds. En cada fold, se entrenó un modelo de regresión logística y se optimizaron sus hiperparametros (C, penalty, class_weights) y se eligió el mejor modelo utilizando la métrica [F1](https://en.wikipedia.org/wiki/F-score).
2. Una vez entrenados los modelos por fold, se generan las predicciones de cada modelo y se elige la predicción definitiva utilizando un proceso de [votación suave](https://machinelearningmastery.com/voting-ensembles-with-python/).

El modelo de análisis de sentimiento no requería ser entrenado.

### 3.3 Otros variantes probadas

Durante el entrenamiento y selección de los modelos predictivos, se probaron otras alternativas a los modelos nombrados anteriormente. Sin embargo, estas fueron descartadas por no mejorar las métricas utilizadas para evaluar los modelos o tener un costo prohibitivo para hacerse a gran escala (en el caso de GPT-3).

Estas fueron las otras alternativas utilizadas:

1. [Bag of words](https://en.wikipedia.org/wiki/Bag-of-words_model) en vez de embeddings de BETO.
2. Otros modelos preentrenados tales como [BERTIN](https://huggingface.co/bertin-project/bertin-roberta-base-spanish) y [MarIA](https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne)
3. Otros algoritmos en vez de regresión logística, tales como [LightGBM](https://en.wikipedia.org/wiki/LightGBM), [Random forest](https://en.wikipedia.org/wiki/Random_forest) o [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine).
4. Ajuste fino de GPT-3. 

## 4. Métricas reportadas

Para todos los modelos se reportan las siguientes métricas:

1. [F1](https://en.wikipedia.org/wiki/F-score) (promedio macro y por clase)
2. [Acierto](https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/)
3. [Kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa)
4. [Precision](https://en.wikipedia.org/wiki/Precision_and_recall) (promedio macro y por clase)
5. [Recall](https://en.wikipedia.org/wiki/Precision_and_recall) (promedio macro y por clase)

Las métricas finales pueden ser consultadas [aquí](https://docs.google.com/spreadsheets/d/1i2ZPUVwsUD0RRSkmvkMIK1DacKq-9WrxIbrsHUbM-_Q/edit#gid=1412364032).

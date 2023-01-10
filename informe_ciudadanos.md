# Proyecto Clasificación de tweets para Univ. Autónoma Madrid

## 1. Introducción

El objetivo principal de este proyecto es clasificar los tweets publicados por ciudadanos en respuesta a tweets publicados por los ayuntamientos españoles en base a una taxonomías determinada, utilizando modelos de Aprendizaje Automático (Machine Learning), para el análisis posterior de dichos tweets.

Para ello se obtuvieron los últimos 3200 tweets publicados por ciudadanos en respuesta o con mención a las cuentas oficiales de los distintitos ayuntamientos españoles a través de la API de twitter, una vez obtenidos los datos se procedió al etiquetado de una muestra representativa de estos para poder entrenar modelos que permita etiquetar el resto de los tweets.

## 2. Extracción y etiquetado de datos

Se extrajeron los últimos tweets publicados por ciudadanos en respuesta o con mención a las cuentas oficiales de los ayuntamientos españoles a través de la API de Twitter. Una vez obtenidos se procedió al etiquetado de una muestra representativa de los mismos utilizando [Doccano](https://github.com/doccano/doccano), para luego crear los modelos de clasificación con dichos datos.

Inicialmente, se extrajeron 584.686 tweets, de los cuáles se excluyeron los tweets que no estaban en castellano, los retweets y los tweets cuyos autores eran ayuntamientos españoles. Lo cuál resultó en 135.998 tweets, a partir de los cuáles se creó una [muestra estratificada](https://es.wikipedia.org/wiki/Muestreo_estratificado) (a nivel de días) de 2.497 tweets, que posteriormente fueron etiquetados manualmente.

Los tweets etiquetados se encuentran disponibles en el fichero `all_citizens_labeled.csv`.

### 2.1 Taxonomía utilizada

Para clasificar los tweets, se entrenó un modelo aprendizaje automático con base en una taxonomía de 5 clases:

- Informativo demandante
- Expresivo
- Destructivo
- Informativo colaborardor
- Entretenido

Adicionalmente, se utilizaron modelos preentrenados para evaluar las siguientes características de los tweets:

1. **Análisis de sentimiento:**  
    - Negativo
    - Neutral
    - Positivo
    
2. **Análisis de emociones:**
    - Joy (Alegría)
    - Sadness (Tristeza)
    - Disgust (Asco)
    - Anger (Rabia)
    - Surprise (Sorpresa)
    - Fear (Miedo)
    - Others (Otras)
    
3. **Análisis de discurso de odio:**  
    - Hateful (Odioso)
    - Targeted (Dirigido)
    - Agressive (Agresivo)
 

## 3. Creación de modelos predictivos

Se creó un modelo de clasificación por cada capa definida en la sección anterior. Para ello, se utilizaron las librerías de Python [scikit-learn](https://scikit-learn.org/stable/) y [transformers](https://huggingface.co/docs/transformers/index), que contienen una gran variedad de algoritmos de aprendizaje automático especializados en procesamiento de lenguaje natural. En concreto, se han utilizado los siguientes algoritmos:

1. Se creó un odelo de regresión logística, combinado con [embeddings](https://huggingface.co/hiiamsid/sentence_similarity_spanish_es) creados a partir de BETO, pero entrenados para evaluar similaridad de texto.

2. Se analizaron los tweets utilizando la librería de Python [pysentimiento](https://huggingface.co/finiteautomata/beto-sentiment-analysis) para extraer características de sentimiento, emociones y discurso de odio.

3. Adicionalmente, se clasificó como `usuario_relacional` cualquier tweet que hiciera mención de otra cuenta de twitter. Es decir, si '@' estaba contenido en el texto. 

### 3.1 Preprocesamiento

Se quitaron los tweets que no estaban en castellano, los retweets y aquellos tweets cuyo autor era un ayuntamiento español. Luego, a nivel del texto, se quitaron enlaces, reemplazaron emojis por palabras y se quitó el símbolo utilizado para referirse a un usuario ("@").

### 3.2 Entrenamiento de modelos

1. Se siguió un proceso de validación cruzada de 10 folds. En cada fold, se entrenó un modelo de regresión logística y se optimizaron sus hiperparametros (C, penalty, class_weights) y se eligió el mejor modelo utilizando la métrica [F1](https://en.wikipedia.org/wiki/F-score) (agregado y por clase).
2. Una vez entrenados los modelos por fold, se generan las predicciones de cada modelo y se elige la predicción definitiva utilizando un proceso de [votación suave](https://machinelearningmastery.com/voting-ensembles-with-python/).
Los modelos de análisis de sentimiento, emociones y discurso de odio no requerían ser entrenados.

### 3.3 Otros variantes probadas

Durante el entrenamiento y selección de los modelos predictivos, se probaron otras alternativas a los modelos nombrados anteriormente. Sin embargo, estas fueron descartadas por tener un rendimiento inferior al modelo seleccionado.

Estas fueron las otras alternativas utilizadas:

1. [Bag of words](https://en.wikipedia.org/wiki/Bag-of-words_model) en vez de embeddings de BETO.
2. Otros algoritmos en vez de regresión logística, tales como [LightGBM](https://en.wikipedia.org/wiki/LightGBM), [Random forest](https://en.wikipedia.org/wiki/Random_forest) o [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine).
3. Modelo [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) haciendo un ajuste fino con los datos disponibles.

## 4. Métricas reportadas

Para todos los modelos se reportan las siguientes métricas:

1. [Accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision)
2. [F1](https://en.wikipedia.org/wiki/F-score) (agregado y por clase)
3. [Kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa)
4. [Precision](https://en.wikipedia.org/wiki/Precision_and_recall) (agregado y por clase)
5. [Recall](https://en.wikipedia.org/wiki/Precision_and_recall) (agregado y por clase)

Las métricas finales pueden ser consultadas [aquí](https://docs.google.com/spreadsheets/d/13HnWomNd2f2YZAKTw5L3jBjAL8K13BFXG_m_TKTDxU4).

Proyecto-PRY20220232

🎓 Descripción del Proyecto

Este proyecto es un modelo predictivo de deserción universitaria desarrollado en Python utilizando algoritmos de regresión logística. El objetivo del proyecto es identificar con antelación a los estudiantes que corren el riesgo de abandonar sus estudios, lo que permite implementar medidas preventivas y apoyo personalizado para disminuir esta probabilidad. 🎯

La arquitectura del proyecto se basa en Azure Data Factory para realizar la extracción, transformación y carga (ETL) de los datos, y Blob Storage para almacenar el dataset. Una vez que los datos han sido procesados y analizados, los resultados de las predicciones se visualizan a través de un dashboard en Power BI, proporcionando una interfaz amigable y fácil de interpretar para los usuarios. 📊

📝 Documento de Despliegue
1. Preparación de los datos: 📦 La primera etapa del proyecto implica la recopilación de los datos y su carga en Azure Blob Storage. Los datos pueden ser de varias fuentes y en diferentes formatos.

2. ETL en Azure Data Factory: 🔄 Los datos almacenados en Blob Storage se transforman y limpian a través de un proceso ETL en Azure Data Factory. Este proceso implica la eliminación de datos no deseados o irrelevantes, la resolución de conflictos de datos y la preparación de los datos para el análisis.

3. Creación del Modelo de Regresión Logística: 📈 En este paso, se desarrolla un modelo de regresión logística en Python para predecir la deserción universitaria. El modelo se entrena con el dataset preparado en el paso anterior.

4. Predicción y Análisis de Resultados: 🔍 Una vez entrenado el modelo, se utiliza para hacer predicciones sobre el conjunto de datos. Estos resultados se analizan para determinar la exactitud del modelo y hacer cualquier ajuste necesario.

5. Visualización de los Resultados: 🖥️ Los resultados de las predicciones se muestran en un dashboard de Power BI. Esta visualización permite a los usuarios interpretar fácilmente los resultados y tomar decisiones basadas en ellos.

Este proyecto se encuentra en constante desarrollo, incorporando nuevas funcionalidades y mejoras con el fin de mejorar su precisión y usabilidad. Te invitamos a colaborar y contribuir a este proyecto para ayudar a prevenir la deserción universitaria. 🤝

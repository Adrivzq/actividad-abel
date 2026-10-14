# Brújula — Orientador académico educativo

Aplicación alternativa inspirada en la estructura educativa de Sign Detector, pero con otro objetivo: explorar estudios y profesiones mediante un perfil de intereses y comparar un **árbol de decisión**, un **Random Forest** y un **sistema experto de reglas**.

## Ejecutar

Desde la carpeta del proyecto y con las dependencias de `requirements.txt` instaladas:

```powershell
.venv\Scripts\python.exe -m streamlit run app_orientador.py
```

La aplicación estará disponible en `http://localhost:8501`.

## Cómo se usa

1. Valora seis áreas de interés de 0 a 5.
2. Abre la pestaña **Orientación** para consultar las rutas sugeridas.
3. Compara las rutas de los dos modelos supervisados con las reglas activadas y revisa las opciones de estudio.
4. Usa **Cómo funciona** para entender las limitaciones y el origen de los perfiles.

## Métodos

- **Árbol de decisión:** modelo supervisado que aprende divisiones y devuelve las proporciones de clases en su hoja final.
- **Random Forest:** conjunto de árboles entrenados con muestreo con reemplazo; muestra la proporción de votos para cada ruta.
- **Sistema experto:** base de conocimiento con reglas `IF/THEN` y un motor de inferencia que suma puntos y explica las señales encontradas.
- **Perfiles de demostración:** el bosque se entrena localmente al iniciar con arquetipos y respuestas simuladas reproducibles. No requiere descargar un dataset, usar cuentas externas ni conservar un archivo de modelo.

Los perfiles de entrenamiento son sintéticos y las rutas se han definido para demostrar los algoritmos, no para medir aptitudes reales. La aplicación no es un test vocacional validado, ni predice el éxito académico o laboral; sus resultados son ideas iniciales para explorar y contrastar con orientación profesional.

## Despliegue en Render

Si quieres publicar esta aplicación como un servicio separado del detector original:

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run app_orientador.py --server.port $PORT --server.address 0.0.0.0`

La configuración de despliegue existente corresponde al detector original y no se modifica.

## Pruebas

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests
```

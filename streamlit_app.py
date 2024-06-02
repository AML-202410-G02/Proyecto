import streamlit as st
import pandas as pd
import numpy as np

from src.back.ModelController import ModelController

st.set_page_config(
    layout="centered", page_title="Make predictions", page_icon="🛩️"
)

### Support functions
def generate_progress_bar(value):
    return f'<div style="width: 100%; border: 1px solid #eee; border-radius: 10px;"><div style="width: {value * 100}%; height: 24px; background: linear-gradient(90deg, rgba(62,149,205,1) 0%, rgba(90,200,250,1) 100%); border-radius: 10px;"></div></div>'

ctrl = ModelController()

# UI
st.title('Predicción de consumo de bateria en drones')
# Texto introductorio
st.markdown("""
Bienvenido a la herramienta. Esta aplicación permite 
estimar el consumo de batería de drones bajo diferentes condiciones de vuelo usando Técnicas supervisadas de Machine Learning. 

## ¿Cómo funciona?
1. Suba un archivo CSV con los datos de vuelo de drones.
2. La aplicación el archivo con los cuales quire hacer la prueba.
3. De clic en 'Hacer predicciones' para que muestre los resultados. Este mostrará para cada prueba de vuelo ingresada el consumo que tendrá la bateria.

Para comenzar, por favor, sube tu archivo CSV utilizando el botón de abajo.
""")
# Subir archivo CSV
uploaded_file = st.file_uploader("Subir archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Datos del archivo CSV:")
    st.write(df.head())

    # Realizar predicciones si hay datos cargados
    if st.button('Hacer predicciones'):
        try:
            predictions = ctrl.predict(df)
            
            # Convertir las predicciones a una serie de Pandas
            predictions_series = pd.Series(predictions, name='Predicción')
            
            # Mapear las predicciones numéricas a etiquetas de texto
            pred_map = {1: 'Bajo', 2: 'Medio', 3: 'Alto'}
            predictions_series = predictions_series.map(pred_map)
            
            # Agregar las predicciones al DataFrame original
            df['Predicción'] = predictions_series
            
            st.subheader('Predicción')
            st.write(df)

            st.success("✅ Done!")

            #st.markdown(result_df.to_html(escape=False), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Something happened: {e}", icon="🚨")

# Importación de bibliotecas necesarias
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

# Inicialización de la aplicación FastAPI
app = FastAPI()

# Definición de la carpeta para guardar las gráficas generadas
GRAPH_PATH = "./generated_graphs"
os.makedirs(GRAPH_PATH, exist_ok=True)  # Crear la carpeta si no existe

# Ruta para subir un archivo CSV y generar gráficas
@app.post("/generate-graphs/")
async def generate_graphs(file: UploadFile):
    # Validar que el archivo tenga extensión CSV
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV.")

    # Intentar leer el archivo CSV
    try:
        data = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo: {e}")

    # Validar que el archivo contenga las columnas esperadas
    required_columns = {"Fecha", "Tipo de Alimento", "Cantidad Consumida (gr)"}
    if not required_columns.issubset(data.columns):
        raise HTTPException(status_code=400, detail=f"El archivo debe contener las columnas: {required_columns}")

    # Procesar los datos
    data['Fecha'] = pd.to_datetime(data['Fecha'])  # Convertir la columna 'Fecha' a tipo datetime
    data.set_index('Fecha', inplace=True)  # Establecer la columna 'Fecha' como índice

    # ---- Generar gráfica de pastel ---- #
    tipo_alimento_totales = data.groupby('Tipo de Alimento')['Cantidad Consumida (gr)'].sum()  # Sumar cantidades por tipo de alimento
    pie_path = os.path.join(GRAPH_PATH, "grafica_pastel.png")  # Definir la ruta para guardar la gráfica
    plt.figure(figsize=(8, 8))  # Configurar tamaño de la gráfica
    plt.pie(tipo_alimento_totales, labels=tipo_alimento_totales.index, autopct='%1.1f%%', startangle=90)  # Crear gráfica de pastel
    plt.title('Distribución del Consumo por Tipo de Alimento')
    plt.savefig(pie_path)  # Guardar la gráfica como imagen
    plt.close()  # Cerrar la gráfica

    # ---- Generar gráfica de consumo diario ---- #
    data = data.asfreq('D', fill_value=0)  # Rellenar fechas faltantes con ceros
    daily_consumption = data['Cantidad Consumida (gr)']  # Obtener los datos de consumo diario
    line_path = os.path.join(GRAPH_PATH, "grafica_linea.png")  # Definir la ruta para guardar la gráfica
    plt.figure(figsize=(12, 6))  # Configurar tamaño de la gráfica
    plt.plot(daily_consumption.index, daily_consumption, label="Consumo Diario", color="blue")  # Crear gráfica de línea
    plt.title("Consumo Diario de Alimentos")
    plt.legend()  # Mostrar leyenda
    plt.savefig(line_path)  # Guardar la gráfica como imagen
    plt.close()  # Cerrar la gráfica

    # ---- Predicciones con SARIMA y Holt-Winters ---- #
    # Modelo SARIMA
    sarima_model = SARIMAX(daily_consumption, order=(1, 1, 1), seasonal_order=(1, 1, 0, 7))  # Configurar modelo SARIMA
    sarima_fit = sarima_model.fit(disp=False)  # Ajustar el modelo
    sarima_forecast = sarima_fit.get_forecast(steps=30).predicted_mean  # Generar predicciones para 30 días

    # Modelo Holt-Winters
    holt_model = ExponentialSmoothing(daily_consumption, trend='add', seasonal='add', seasonal_periods=7).fit()  # Configurar modelo Holt-Winters
    holt_forecast = holt_model.forecast(steps=30)  # Generar predicciones para 30 días

    # Generar gráfica de predicciones
    pred_path = os.path.join(GRAPH_PATH, "grafica_predicciones.png")  # Definir la ruta para guardar la gráfica
    plt.figure(figsize=(14, 8))  # Configurar tamaño de la gráfica
    plt.plot(daily_consumption.index, daily_consumption, label="Datos Reales", color="blue")  # Gráfica de datos reales
    plt.plot(sarima_forecast.index, sarima_forecast, label="Predicción SARIMA", color="red", linestyle="--")  # Predicción SARIMA
    plt.plot(holt_forecast.index, holt_forecast, label="Predicción Holt-Winters", color="green", linestyle="--")  # Predicción Holt-Winters
    plt.title("Predicciones del Consumo Diario")
    plt.legend()  # Mostrar leyenda
    plt.savefig(pred_path)  # Guardar la gráfica como imagen
    plt.close()  # Cerrar la gráfica

    # Devolver las rutas de las gráficas generadas
    return {
        "pie_chart": pie_path,
        "line_chart": line_path,
        "prediction_chart": pred_path
    }

# Ruta para descargar una gráfica generada
@app.get("/download-graph/")
async def download_graph(filename: str):
    filepath = os.path.join(GRAPH_PATH, filename)  # Construir la ruta completa del archivo
    if not os.path.exists(filepath):  # Verificar si el archivo existe
        raise HTTPException(status_code=404, detail="Gráfica no encontrada.")
    return FileResponse(filepath)  # Devolver el archivo como respuesta


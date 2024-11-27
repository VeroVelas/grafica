import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Cargar los datos
data = pd.read_csv('consumo_mascotas_sin_gasto.csv')
data['Fecha'] = pd.to_datetime(data['Fecha'])
data.set_index('Fecha', inplace=True)

# ---- Gráfica de Pastel: Tipo de Alimento ---- #
# Sumar el consumo total por tipo de alimento
tipo_alimento_totales = data.groupby('Tipo de Alimento')['Cantidad Consumida (gr)'].sum()

plt.figure(figsize=(8, 8))
plt.pie(tipo_alimento_totales, labels=tipo_alimento_totales.index, autopct='%1.1f%%',
        startangle=90, colors=['#FFD700', '#90EE90', '#FF6347'])
plt.title('Distribución del Consumo por Tipo de Alimento', fontsize=16, fontweight='bold')
plt.legend(tipo_alimento_totales.index, title='Tipos de Alimento', loc='upper right', bbox_to_anchor=(1.3, 0.9))
plt.tight_layout()
plt.savefig('grafica_pastel_tipo_alimento.png')  # Guardar gráfica
plt.show()

# ---- Preparación para Series de Tiempo ---- #
# Verificar duplicados en el índice
duplicates = data.index[data.index.duplicated()]
if not duplicates.empty:
    print(f"Fechas duplicadas:\n{duplicates}")
    # Consolidar duplicados sumando los valores
    data = data.groupby(data.index).sum()

# Rellenar días faltantes con ceros
data = data.asfreq('D', fill_value=0)

# Agregar consumo diario
daily_consumption = data['Cantidad Consumida (gr)']

# Suavizar la serie para mostrar tendencias
smoothed_consumption = gaussian_filter1d(daily_consumption, sigma=3)

# ---- Gráfica de Líneas: Consumo Diario ---- #
plt.figure(figsize=(14, 8))
plt.plot(daily_consumption.index, daily_consumption, label='Consumo Diario', color='blue', linewidth=1)
plt.plot(daily_consumption.index, smoothed_consumption, label='Tendencia Suavizada', color='orange', linewidth=2)
plt.title('Consumo Diario de Alimentos', fontsize=16)
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Cantidad Consumida (gr)', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('grafica_lineas_consumo_diario.png')  # Guardar gráfica
plt.show()

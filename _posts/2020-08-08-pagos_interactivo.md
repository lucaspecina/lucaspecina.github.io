---
title: "pagos_por_examen"
date: 2020-08-08
tags: [cons_magist]
---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import ipywidgets as widgets
```

    /Users/lucaspecina/anaconda3/lib/python3.7/site-packages/statsmodels/tools/_testing.py:19: FutureWarning:
    
    pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
    



```python
params = pd.read_csv('/Users/lucaspecina/Desktop/Data/Planificacion Estrategica/Pagos jurados consejo magist/pruebas lineales/NUEVO2020_simulaciones_CONVECTORES_bootstrap_version2.csv')

params['diferencia_limite_boot'] = params.apply(lambda x: \
      [int(valor) for valor in x.diferencia_limite_boot.replace('[','').replace(']','').replace('\n','').split(' ') if valor != ''],
                                               axis='columns')
params['MITAD_MAS_diferencia_limite_boot'] = params.apply(lambda x: \
      [int(valor) for valor in x.MITAD_MAS_diferencia_limite_boot.replace('[','').replace(']','').replace('\n','').split(' ') if valor != ''],
                                               axis='columns')
params['DOBLE_diferencia_limite_boot'] = params.apply(lambda x: \
      [int(valor) for valor in x.DOBLE_diferencia_limite_boot.replace('[','').replace(']','').replace('\n','').split(' ') if valor != ''],
                                               axis='columns')
```


```python
def plotear(piso_w=8_000,pend_w=100,tope_w=60_000):
    data_plot = params[(params.piso==piso_w) & (params.pend==pend_w) & (params.tope==tope_w)]
    
    fig,axes = plt.subplots(3,1,figsize=(14,11),sharex=True)
    sns.distplot(data_plot.diferencia_limite_boot.values[0],ax=axes[0])
    axes[0].axvline(x=0,color='red',linestyle='--')
    axes[0].set_xlim([-1_500_000,1_500_000])
    axes[0].set_title('Simulaciones con misma cantidad de concursos que 2020')
    sns.distplot(data_plot.MITAD_MAS_diferencia_limite_boot.values[0],ax=axes[1])
    axes[1].axvline(x=0,color='red',linestyle='--')
    axes[1].set_title('Simulaciones con %50 más de concursos que 2020')
    sns.distplot(data_plot.DOBLE_diferencia_limite_boot.values[0],ax=axes[2])
    plt.axvline(x=0,color='red',linestyle='--')
    axes[2].set_title('Simulaciones con el doble de concursos que 2020')
    plt.text(x=-100_000,y=0,s='límite presupuesto',color='red',fontsize=14)
    
#     print(f'Con los parámetros:\n Piso: {piso_w}\n Pendiente: {pend_w}\n Tope: {tope_w}')
    print(f'\nRESULTADOS (Si da positivo es que hay un ahorro. Si es negativo es porque supera el límite)')
    print(f'\nSimulacion con misma cantidad de concursos que 2020:\n- Promedio(comparación con límite presupuestario): ${data_plot.media_boot.values[0]}')
    print(f'- Desvío Estándar: ${data_plot.std_boot.values[0]}\n- Porcentaje de las simulaciones que no superan límite: %{data_plot.porcentaje_inferior_0.values[0]*100}')

    print(f'\nSimulacion con %50 mas de concursos que 2020:\n- Promedio(comparación con límite presupuestario): ${data_plot.MITAD_MAS_media_boot.values[0]}')
    print(f'- Desvío Estándar: ${data_plot.MITAD_MAS_std_boot.values[0]}\n- Porcentaje de las simulaciones que no superan límite: %{data_plot.MITAD_MAS_porcentaje_inferior_0.values[0]*100}')

    print(f'\nSimulacion con el doble de concursos que 2020:\n- Promedio(comparación con límite presupuestario): ${data_plot.DOBLE_media_boot.values[0]}')
    print(f'- Desvío Estándar: ${data_plot.DOBLE_std_boot.values[0]}\n- Porcentaje de las simulaciones que no superan límite: %{data_plot.DOBLE_porcentaje_inferior_0.values[0]*100}')

    print(f'\nComparación con el método anterior ($31200 por corrector): ${data_plot.diferencia_metodo.values[0]}')
    #     return data_plot.drop(columns=['diferencia_limite_boot','MITAD_MAS_diferencia_limite_boot','DOBLE_diferencia_limite_boot'])

slider = widgets.interact(plotear, piso_w=[8_000,12_000,16_000],pend_w = [100,150,200,250,300], tope_w = [60_000,75_000])
display(slider)
```


    interactive(children=(Dropdown(description='piso_w', options=(8000, 12000, 16000), value=8000), Dropdown(descr…



    <function __main__.plotear(piso_w=8000, pend_w=100, tope_w=60000)>



```python

```


```python

```


```python

```

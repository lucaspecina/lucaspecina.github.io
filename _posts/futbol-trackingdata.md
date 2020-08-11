---
title: "tracking futbol"
date: 2020-08-11
tags: [data science, football]
header:
  image: "/images/pruebas_futbol/output_68_1.png"
excerpt: "Pruebas futbol"
mathjax: "true"
---

https://github.com/metrica-sports/sample-data data de tracking - metrica sports

https://www.youtube.com/watch?v=8TrleFklEsE friends of tracking- video explicativo de como comenzar a rtabajar con tracking data

https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking codigo de esos videos

https://github.com/rjtavares/football-crunching/blob/master/notebooks/working%20with%20positional%20data.ipynb como dibujar cancha



```python
%matplotlib notebook
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import networkx as nx
import ipywidgets as widgets
```


```python

```

# Events Data


```python
events_game1= pd.read_csv('/Users/lucaspecina/Desktop/Data/Otros/futbol/sample-data-master/data/Sample_Game_1/Sample_Game_1_RawEventsData.csv')
events_game1.head()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Type</th>
      <th>Subtype</th>
      <th>Period</th>
      <th>Start Frame</th>
      <th>Start Time [s]</th>
      <th>End Frame</th>
      <th>End Time [s]</th>
      <th>From</th>
      <th>To</th>
      <th>Start X</th>
      <th>Start Y</th>
      <th>End X</th>
      <th>End Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Away</td>
      <td>SET PIECE</td>
      <td>KICK OFF</td>
      <td>1</td>
      <td>1</td>
      <td>0.04</td>
      <td>0</td>
      <td>0.00</td>
      <td>Player19</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Away</td>
      <td>PASS</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0.04</td>
      <td>3</td>
      <td>0.12</td>
      <td>Player19</td>
      <td>Player21</td>
      <td>0.45</td>
      <td>0.39</td>
      <td>0.55</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Away</td>
      <td>PASS</td>
      <td>NaN</td>
      <td>1</td>
      <td>3</td>
      <td>0.12</td>
      <td>17</td>
      <td>0.68</td>
      <td>Player21</td>
      <td>Player15</td>
      <td>0.55</td>
      <td>0.43</td>
      <td>0.58</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Away</td>
      <td>PASS</td>
      <td>NaN</td>
      <td>1</td>
      <td>45</td>
      <td>1.80</td>
      <td>61</td>
      <td>2.44</td>
      <td>Player15</td>
      <td>Player19</td>
      <td>0.55</td>
      <td>0.19</td>
      <td>0.45</td>
      <td>0.31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Away</td>
      <td>PASS</td>
      <td>NaN</td>
      <td>1</td>
      <td>77</td>
      <td>3.08</td>
      <td>96</td>
      <td>3.84</td>
      <td>Player19</td>
      <td>Player21</td>
      <td>0.45</td>
      <td>0.32</td>
      <td>0.49</td>
      <td>0.47</td>
    </tr>
  </tbody>
</table>
</div>




```python
events_game1.groupby('Team').Type.value_counts()
```




    Team  Type          
    Away  PASS              362
          RECOVERY          143
          BALL LOST         128
          CHALLENGE         115
          BALL OUT           33
          SET PIECE          32
          FAULT RECEIVED      7
          SHOT                6
          CARD                2
    Home  PASS              437
          RECOVERY          135
          BALL LOST         129
          CHALLENGE         118
          SET PIECE          45
          BALL OUT           18
          SHOT               18
          FAULT RECEIVED     15
          CARD                2
    Name: Type, dtype: int64




```python
from Metrica_IO import to_metric_coordinates

events_game1 = to_metric_coordinates(events_game1)
```


```python
# GOLES
events_game1[(events_game1.Subtype.str.contains('-GOAL',na=False))]\
[['Team','Subtype','From','End Time [s]']].assign(Minute= lambda x: x['End Time [s]']/60)\
[['Team','Subtype','From','Minute']].rename(columns={\
                            'Team':'Equipo','Subtype':'Tipo de Gol','From':'Goleador','Minute':'Minuto'})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Equipo</th>
      <th>Tipo de Gol</th>
      <th>Goleador</th>
      <th>Minuto</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <td>Home</td>
      <td>HEAD-ON TARGET-GOAL</td>
      <td>Player9</td>
      <td>1.539333</td>
    </tr>
    <tr>
      <th>1093</th>
      <td>Home</td>
      <td>WOODWORK-GOAL</td>
      <td>Player5</td>
      <td>58.323333</td>
    </tr>
    <tr>
      <th>1114</th>
      <td>Home</td>
      <td>ON TARGET-GOAL</td>
      <td>Player10</td>
      <td>60.017333</td>
    </tr>
    <tr>
      <th>1213</th>
      <td>Home</td>
      <td>ON TARGET-GOAL</td>
      <td>Player9</td>
      <td>66.030667</td>
    </tr>
  </tbody>
</table>
</div>




```python
from IPython.display import display
```


```python
from Metrica_Viz import plot_pitch
fig,ax= plot_pitch()
```


![png](output_9_0.png)


Con el fig,ax se le puede agregar mas plots encima


```python
ax.plot(events_game1.loc[34]['Start X'], events_game1.loc[34]['Start Y'], 'ro')
display(fig)
```


![png](output_11_0.png)



```python
# le agrego una linea hacia el final
ax.annotate("", xy=events_game1.loc[34][['End X','End Y']], xytext=events_game1.loc[34][['Start X','Start Y']]\
            , alpha=0.6, arrowprops=dict(arrowstyle="->",color='r'))
display(fig)
```


![png](output_12_0.png)



```python
from Metrica_Viz import plot_events

plot_events(events_game1.loc[32:34], indicators= ['Marker','Arrow'],figax=(fig,ax), annotate=True)
display(fig)
```


![png](output_13_0.png)



```python

```

# Tracking Data


```python
from Metrica_IO import tracking_data

??tracking_data
```


    [0;31mSignature:[0m [0mtracking_data[0m[0;34m([0m[0mDATADIR[0m[0;34m,[0m [0mgame_id[0m[0;34m,[0m [0mteamname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mSource:[0m   
    [0;32mdef[0m [0mtracking_data[0m[0;34m([0m[0mDATADIR[0m[0;34m,[0m[0mgame_id[0m[0;34m,[0m[0mteamname[0m[0;34m)[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m    [0;34m'''[0m
    [0;34m    tracking_data(DATADIR,game_id,teamname):[0m
    [0;34m    read Metrica tracking data for game_id and return as a DataFrame. [0m
    [0;34m    teamname is the name of the team in the filename. For the sample data this is either 'Home' or 'Away'.[0m
    [0;34m    '''[0m[0;34m[0m
    [0;34m[0m    [0mteamfile[0m [0;34m=[0m [0;34m'/Sample_Game_%d/Sample_Game_%d_RawTrackingData_%s_Team.csv'[0m [0;34m%[0m [0;34m([0m[0mgame_id[0m[0;34m,[0m[0mgame_id[0m[0;34m,[0m[0mteamname[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m    [0;31m# First:  deal with file headers so that we can get the player names correct[0m[0;34m[0m
    [0;34m[0m    [0mcsvfile[0m [0;34m=[0m  [0mopen[0m[0;34m([0m[0;34m'{}/{}'[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mDATADIR[0m[0;34m,[0m [0mteamfile[0m[0;34m)[0m[0;34m,[0m [0;34m'r'[0m[0;34m)[0m [0;31m# create a csv file reader[0m[0;34m[0m
    [0;34m[0m    [0mreader[0m [0;34m=[0m [0mcsv[0m[0;34m.[0m[0mreader[0m[0;34m([0m[0mcsvfile[0m[0;34m)[0m [0;34m[0m
    [0;34m[0m    [0mteamnamefull[0m [0;34m=[0m [0mnext[0m[0;34m([0m[0mreader[0m[0;34m)[0m[0;34m[[0m[0;36m3[0m[0;34m][0m[0;34m.[0m[0mlower[0m[0;34m([0m[0;34m)[0m[0;34m[0m
    [0;34m[0m    [0mprint[0m[0;34m([0m[0;34m"Reading team: %s"[0m [0;34m%[0m [0mteamnamefull[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m    [0;31m# construct column names[0m[0;34m[0m
    [0;34m[0m    [0mjerseys[0m [0;34m=[0m [0;34m[[0m[0mx[0m [0;32mfor[0m [0mx[0m [0;32min[0m [0mnext[0m[0;34m([0m[0mreader[0m[0;34m)[0m [0;32mif[0m [0mx[0m [0;34m!=[0m [0;34m''[0m[0;34m][0m [0;31m# extract player jersey numbers from second row[0m[0;34m[0m
    [0;34m[0m    [0mcolumns[0m [0;34m=[0m [0mnext[0m[0;34m([0m[0mreader[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m    [0;32mfor[0m [0mi[0m[0;34m,[0m [0mj[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0mjerseys[0m[0;34m)[0m[0;34m:[0m [0;31m# create x & y position column headers for each player[0m[0;34m[0m
    [0;34m[0m        [0mcolumns[0m[0;34m[[0m[0mi[0m[0;34m*[0m[0;36m2[0m[0;34m+[0m[0;36m3[0m[0;34m][0m [0;34m=[0m [0;34m"{}_{}_x"[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mteamname[0m[0;34m,[0m [0mj[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m        [0mcolumns[0m[0;34m[[0m[0mi[0m[0;34m*[0m[0;36m2[0m[0;34m+[0m[0;36m4[0m[0;34m][0m [0;34m=[0m [0;34m"{}_{}_y"[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mteamname[0m[0;34m,[0m [0mj[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m    [0mcolumns[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m [0;34m=[0m [0;34m"ball_x"[0m [0;31m# column headers for the x & y positions of the ball[0m[0;34m[0m
    [0;34m[0m    [0mcolumns[0m[0;34m[[0m[0;34m-[0m[0;36m1[0m[0;34m][0m [0;34m=[0m [0;34m"ball_y"[0m[0;34m[0m
    [0;34m[0m    [0;31m# Second: read in tracking data and place into pandas Dataframe[0m[0;34m[0m
    [0;34m[0m    [0mtracking[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34m'{}/{}'[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mDATADIR[0m[0;34m,[0m [0mteamfile[0m[0;34m)[0m[0;34m,[0m [0mnames[0m[0;34m=[0m[0mcolumns[0m[0;34m,[0m [0mindex_col[0m[0;34m=[0m[0;34m'Frame'[0m[0;34m,[0m [0mskiprows[0m[0;34m=[0m[0;36m3[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m    [0;32mreturn[0m [0mtracking[0m[0;34m[0m[0;34m[0m[0m
    [0;31mFile:[0m      ~/Desktop/Data/Otros/futbol/Metrica_IO.py
    [0;31mType:[0m      function




```python
track_home_g1= tracking_data('/Users/lucaspecina/Desktop/Data/Otros/futbol/sample-data-master/data',1,'Home')
track_away_g1= tracking_data('/Users/lucaspecina/Desktop/Data/Otros/futbol/sample-data-master/data',1,'Away')
# NICE
track_home_g1.head()
```

    Reading team: home
    Reading team: away





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Period</th>
      <th>Time [s]</th>
      <th>Home_11_x</th>
      <th>Home_11_y</th>
      <th>Home_1_x</th>
      <th>Home_1_y</th>
      <th>Home_2_x</th>
      <th>Home_2_y</th>
      <th>Home_3_x</th>
      <th>Home_3_y</th>
      <th>...</th>
      <th>Home_10_x</th>
      <th>Home_10_y</th>
      <th>Home_12_x</th>
      <th>Home_12_y</th>
      <th>Home_13_x</th>
      <th>Home_13_y</th>
      <th>Home_14_x</th>
      <th>Home_14_y</th>
      <th>ball_x</th>
      <th>ball_y</th>
    </tr>
    <tr>
      <th>Frame</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.04</td>
      <td>0.00082</td>
      <td>0.48238</td>
      <td>0.32648</td>
      <td>0.65322</td>
      <td>0.33701</td>
      <td>0.48863</td>
      <td>0.30927</td>
      <td>0.35529</td>
      <td>...</td>
      <td>0.55243</td>
      <td>0.43269</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.45472</td>
      <td>0.38709</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.08</td>
      <td>0.00096</td>
      <td>0.48238</td>
      <td>0.32648</td>
      <td>0.65322</td>
      <td>0.33701</td>
      <td>0.48863</td>
      <td>0.30927</td>
      <td>0.35529</td>
      <td>...</td>
      <td>0.55243</td>
      <td>0.43269</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.49645</td>
      <td>0.40656</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.12</td>
      <td>0.00114</td>
      <td>0.48238</td>
      <td>0.32648</td>
      <td>0.65322</td>
      <td>0.33701</td>
      <td>0.48863</td>
      <td>0.30927</td>
      <td>0.35529</td>
      <td>...</td>
      <td>0.55243</td>
      <td>0.43269</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.53716</td>
      <td>0.42556</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.16</td>
      <td>0.00121</td>
      <td>0.48238</td>
      <td>0.32622</td>
      <td>0.65317</td>
      <td>0.33687</td>
      <td>0.48988</td>
      <td>0.30944</td>
      <td>0.35554</td>
      <td>...</td>
      <td>0.55236</td>
      <td>0.43313</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.55346</td>
      <td>0.42231</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0.20</td>
      <td>0.00129</td>
      <td>0.48238</td>
      <td>0.32597</td>
      <td>0.65269</td>
      <td>0.33664</td>
      <td>0.49018</td>
      <td>0.30948</td>
      <td>0.35528</td>
      <td>...</td>
      <td>0.55202</td>
      <td>0.43311</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.55512</td>
      <td>0.40570</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
track_home_g1 = to_metric_coordinates(track_home_g1)
track_away_g1 = to_metric_coordinates(track_away_g1)
```


```python
HOME= track_home_g1.iloc[range(0,500,30)]
AWAY= track_away_g1.iloc[range(0,500,30)]
```


```python
dict_game = {}
for i in range(10):
    plot= plot_frame(HOME.iloc[i:i+1], AWAY.iloc[i:i+1])
    dict_game[str(i)]= plot
```


![png](output_20_0.png)



![png](output_20_1.png)



![png](output_20_2.png)



![png](output_20_3.png)



![png](output_20_4.png)



![png](output_20_5.png)



![png](output_20_6.png)



![png](output_20_7.png)



![png](output_20_8.png)



![png](output_20_9.png)



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python
# Plot some player trajectories (players 11,1,2,3,4)
fig,ax = plot_pitch()
```


![png](output_31_0.png)



```python
# JUGADOR 11 EN LOS PRIMEROS 60 SEGUNDOS (ES EL ARQUERO)
ax.plot( track_home_g1['Home_11_x'].iloc[:1500], track_home_g1['Home_11_y'].iloc[:1500], 'r.', MarkerSize=1)
display(fig)
```


![png](output_32_0.png)



```python
# MOVIMIENTO DE LOS 11 JUGADORES LOCALES EN LOS PRIMEROS 250/25 = 10 SEGUNDOS

for jug in range(11):
    ax.plot( track_home_g1['Home_'+str(jug+1)+'_x'].iloc[:250], track_home_g1['Home_'+str(jug+1)+'_y'].iloc[:250], MarkerSize=1)

display(fig)
```


![png](output_33_0.png)



```python
for away in range(15,26):
    ax.plot( track_away_g1['Away_'+str(away)+'_x'].iloc[:250], track_away_g1['Away_'+str(away)+'_y'].iloc[:250], MarkerSize=1)

display(fig)
```


![png](output_34_0.png)



```python
# JUGADOR 25 EN LOS PRIMEROS 60 SEGUNDOS (ES EL ARQUERO)
ax.plot( track_away_g1['Away_18_x'].iloc[:500], track_away_g1['Away_18_y'].iloc[:500], MarkerSize=1)
display(fig)
```


![png](output_35_0.png)



```python
fig,ax = plot_pitch()
```


![png](output_36_0.png)



```python
# plot player positions at ,atckick-off
from Metrica_Viz import plot_frame

KO_Frame = events_game1.loc[1114]['Start Frame']
fig,ax = plot_frame( track_home_g1.loc[KO_Frame], track_away_g1.loc[KO_Frame] )

plot_events(events_game1.loc[1114:1114], indicators= ['Marker','Arrow'],figax=(fig,ax), annotate=True)


```




    (<Figure size 864x576 with 1 Axes>,
     <matplotlib.axes._subplots.AxesSubplot at 0x1489660b8>)




![png](output_37_1.png)



```python
from IPython.display import Image
Image(filename= '/Users/lucaspecina/Desktop/Lucas/Capturas de pantalla/IMG_D3BABB2661FD-1.jpeg')
```




![jpeg](output_38_0.jpeg)




```python
# JUGADOR 25 EN LOS PRIMEROS 60 SEGUNDOS (ES EL ARQUERO)
ax.plot( track_away_g1['Away_18_x'].iloc[:500], track_away_g1['Away_18_y'].iloc[:500], MarkerSize=1)
display(fig)
```


![png](output_39_0.png)


**Hacerlo con plotly**


```python
track_away_g1.columns
```




    Index(['Period', 'Time [s]', 'Away_25_x', 'Away_25_y', 'Away_15_x',
           'Away_15_y', 'Away_16_x', 'Away_16_y', 'Away_17_x', 'Away_17_y',
           'Away_18_x', 'Away_18_y', 'Away_19_x', 'Away_19_y', 'Away_20_x',
           'Away_20_y', 'Away_21_x', 'Away_21_y', 'Away_22_x', 'Away_22_y',
           'Away_23_x', 'Away_23_y', 'Away_24_x', 'Away_24_y', 'Away_26_x',
           'Away_26_y', 'Away_27_x', 'Away_27_y', 'Away_28_x', 'Away_28_y',
           'ball_x', 'ball_y'],
          dtype='object')




```python
track_home_g1.columns
```




    Index(['Period', 'Time [s]', 'Home_11_x', 'Home_11_y', 'Home_1_x', 'Home_1_y',
           'Home_2_x', 'Home_2_y', 'Home_3_x', 'Home_3_y', 'Home_4_x', 'Home_4_y',
           'Home_5_x', 'Home_5_y', 'Home_6_x', 'Home_6_y', 'Home_7_x', 'Home_7_y',
           'Home_8_x', 'Home_8_y', 'Home_9_x', 'Home_9_y', 'Home_10_x',
           'Home_10_y', 'Home_12_x', 'Home_12_y', 'Home_13_x', 'Home_13_y',
           'Home_14_x', 'Home_14_y', 'ball_x', 'ball_y'],
          dtype='object')




```python
track_away_g1.reset_index().groupby('Period').Frame.first()
```




    Period
    1        1
    2    71269
    Name: Frame, dtype: int64




```python
track_ball_g1= track_away_g1.reset_index().iloc[range(0,500,30)][['Frame','ball_x','ball_y']]
track_ball_g1['equipo']='ball'
track_ball_g1['jugador']='ball'
track_ball_g1= track_ball_g1[['Frame','equipo','jugador','ball_x','ball_y']]
track_ball_g1.columns= ['Frame','equipo','jugador','X','Y']
track_ball_g1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Frame</th>
      <th>equipo</th>
      <th>jugador</th>
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>ball</td>
      <td>ball</td>
      <td>-4.79968</td>
      <td>-7.67788</td>
    </tr>
    <tr>
      <th>30</th>
      <td>31</td>
      <td>ball</td>
      <td>ball</td>
      <td>6.59744</td>
      <td>-20.64140</td>
    </tr>
    <tr>
      <th>60</th>
      <td>61</td>
      <td>ball</td>
      <td>ball</td>
      <td>-4.93854</td>
      <td>-13.31712</td>
    </tr>
    <tr>
      <th>90</th>
      <td>91</td>
      <td>ball</td>
      <td>ball</td>
      <td>-2.45284</td>
      <td>-4.65664</td>
    </tr>
    <tr>
      <th>120</th>
      <td>121</td>
      <td>ball</td>
      <td>ball</td>
      <td>-2.11258</td>
      <td>2.31064</td>
    </tr>
  </tbody>
</table>
</div>




```python
track_away_g1['equipo']= 'away'
track_prueba_away= track_away_g1.iloc[range(0,500,30)].reset_index()[['Frame','Away_15_x',
       'Away_15_y', 'Away_16_x', 'Away_16_y', 'Away_17_x', 'Away_17_y',
       'Away_18_x', 'Away_18_y', 'Away_19_x', 'Away_19_y', 'Away_20_x',
       'Away_20_y', 'Away_21_x', 'Away_21_y', 'Away_22_x', 'Away_22_y',
       'Away_23_x', 'Away_23_y', 'Away_24_x', 'Away_24_y', 'Away_25_x', 'Away_25_y','ball_x','ball_y','equipo']]
track_prueba_away.columns= ['Frame','X-15','Y-15','X-16','Y-16','X-17','Y-17','X-18','Y-18',
                      'X-19','Y-19','X-20','Y-20','X-21','Y-21','X-22','Y-22','X-23','Y-23',
                      'X-24','Y-24','X-25','Y-25','ball_x','ball_y','equipo']
len(track_prueba_away)
```




    17




```python
track_home_g1['equipo']= 'home'
track_prueba_home= track_home_g1.iloc[range(0,500,30)].reset_index()[['Frame','Home_11_x', 'Home_11_y', 
                                                                    'Home_1_x', 'Home_1_y',
       'Home_2_x', 'Home_2_y', 'Home_3_x', 'Home_3_y', 'Home_4_x', 'Home_4_y',
       'Home_5_x', 'Home_5_y', 'Home_6_x', 'Home_6_y', 'Home_7_x', 'Home_7_y',
       'Home_8_x', 'Home_8_y', 'Home_9_x', 'Home_9_y', 'Home_10_x',
       'Home_10_y','ball_x','ball_y','equipo']]
track_prueba_home.columns= ['Frame','X-11','Y-11','X-1','Y-1','X-2','Y-2','X-3','Y-3',
                      'X-4','Y-4','X-5','Y-5','X-6','Y-6','X-7','Y-7','X-8','Y-8',
                      'X-9','Y-9','X-10','Y-10','ball_x','ball_y','equipo']
len(track_prueba_home)
```




    17




```python
# hacer el wide_to_long primero
'''En WIDE_TO_LONG se requieren numeros al final (no se puede hacer al reves (primero numeros y desp axis))'''

track_prueba_away= pd.wide_to_long(track_prueba_away, ['X', 'Y'], i=['Frame','equipo'],
                j='jugador', sep='-').reset_index()
track_prueba_away.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Frame</th>
      <th>equipo</th>
      <th>jugador</th>
      <th>ball_y</th>
      <th>ball_x</th>
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>away</td>
      <td>15</td>
      <td>-7.67788</td>
      <td>-4.79968</td>
      <td>8.89658</td>
      <td>-19.86008</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>away</td>
      <td>16</td>
      <td>-7.67788</td>
      <td>-4.79968</td>
      <td>18.71748</td>
      <td>-2.23720</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>away</td>
      <td>17</td>
      <td>-7.67788</td>
      <td>-4.79968</td>
      <td>18.34860</td>
      <td>18.00368</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>away</td>
      <td>18</td>
      <td>-7.67788</td>
      <td>-4.79968</td>
      <td>-9.77002</td>
      <td>7.83700</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>away</td>
      <td>19</td>
      <td>-7.67788</td>
      <td>-4.79968</td>
      <td>-4.79968</td>
      <td>-7.67788</td>
    </tr>
  </tbody>
</table>
</div>




```python
track_prueba_home= pd.wide_to_long(track_prueba_home, ['X', 'Y'], i=['Frame','equipo'],
                j='jugador', sep='-').reset_index()
track_prueba_home.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Frame</th>
      <th>equipo</th>
      <th>jugador</th>
      <th>ball_y</th>
      <th>ball_x</th>
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>home</td>
      <td>11</td>
      <td>-7.67788</td>
      <td>-4.79968</td>
      <td>-52.91308</td>
      <td>-1.19816</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>home</td>
      <td>1</td>
      <td>-7.67788</td>
      <td>-4.79968</td>
      <td>-18.39312</td>
      <td>10.41896</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>home</td>
      <td>2</td>
      <td>-7.67788</td>
      <td>-4.79968</td>
      <td>-17.27694</td>
      <td>-0.77316</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>home</td>
      <td>3</td>
      <td>-7.67788</td>
      <td>-4.79968</td>
      <td>-20.21738</td>
      <td>-9.84028</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>home</td>
      <td>4</td>
      <td>-7.67788</td>
      <td>-4.79968</td>
      <td>-18.93478</td>
      <td>-19.54184</td>
    </tr>
  </tbody>
</table>
</div>




```python
# hacer el concat

track_prueba= pd.concat([track_prueba_away,track_prueba_home],axis=0)
track_prueba.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Frame</th>
      <th>equipo</th>
      <th>jugador</th>
      <th>ball_y</th>
      <th>ball_x</th>
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>away</td>
      <td>15</td>
      <td>-7.67788</td>
      <td>-4.79968</td>
      <td>8.89658</td>
      <td>-19.86008</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>away</td>
      <td>16</td>
      <td>-7.67788</td>
      <td>-4.79968</td>
      <td>18.71748</td>
      <td>-2.23720</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>away</td>
      <td>17</td>
      <td>-7.67788</td>
      <td>-4.79968</td>
      <td>18.34860</td>
      <td>18.00368</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>away</td>
      <td>18</td>
      <td>-7.67788</td>
      <td>-4.79968</td>
      <td>-9.77002</td>
      <td>7.83700</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>away</td>
      <td>19</td>
      <td>-7.67788</td>
      <td>-4.79968</td>
      <td>-4.79968</td>
      <td>-7.67788</td>
    </tr>
  </tbody>
</table>
</div>




```python
# concat con pelota

track_prueba= track_prueba[['Frame','equipo','jugador','X','Y']]
track_todos= pd.concat([track_prueba,track_ball_g1],axis=0)
track_todos.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Frame</th>
      <th>equipo</th>
      <th>jugador</th>
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>away</td>
      <td>15</td>
      <td>8.89658</td>
      <td>-19.86008</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>away</td>
      <td>16</td>
      <td>18.71748</td>
      <td>-2.23720</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>away</td>
      <td>17</td>
      <td>18.34860</td>
      <td>18.00368</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>away</td>
      <td>18</td>
      <td>-9.77002</td>
      <td>7.83700</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>away</td>
      <td>19</td>
      <td>-4.79968</td>
      <td>-7.67788</td>
    </tr>
  </tbody>
</table>
</div>




```python
#https://plotly.com/python/animations/
#https://www.youtube.com/watch?v=Ercd-Ip5PfQ
#%pylab qt  # wx, gtk, osx or tk
'''px.scatter(track_todos, x='X',y='Y',animation_frame='Frame',animation_group='jugador',
           color='equipo', range_x=[-53,53], range_y=[-34,34])'''
```




    "px.scatter(track_todos, x='X',y='Y',animation_frame='Frame',animation_group='jugador',\n           color='equipo', range_x=[-53,53], range_y=[-34,34])"




```python

```

## Clase 5 : video de partido y velocidades


```python
import Metrica_IO as mio
import Metrica_Viz as mviz
import Metrica_Velocities as mvel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```


```python
events_game1= pd.read_csv('/Users/lucaspecina/Desktop/Data/projj/projj1- futbol/Tracking data - Metrica/data/Sample_Game_1/Sample_Game_1_RawEventsData.csv')
events_game1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Type</th>
      <th>Subtype</th>
      <th>Period</th>
      <th>Start Frame</th>
      <th>Start Time [s]</th>
      <th>End Frame</th>
      <th>End Time [s]</th>
      <th>From</th>
      <th>To</th>
      <th>Start X</th>
      <th>Start Y</th>
      <th>End X</th>
      <th>End Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Away</td>
      <td>SET PIECE</td>
      <td>KICK OFF</td>
      <td>1</td>
      <td>1</td>
      <td>0.04</td>
      <td>0</td>
      <td>0.00</td>
      <td>Player19</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Away</td>
      <td>PASS</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0.04</td>
      <td>3</td>
      <td>0.12</td>
      <td>Player19</td>
      <td>Player21</td>
      <td>0.45</td>
      <td>0.39</td>
      <td>0.55</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Away</td>
      <td>PASS</td>
      <td>NaN</td>
      <td>1</td>
      <td>3</td>
      <td>0.12</td>
      <td>17</td>
      <td>0.68</td>
      <td>Player21</td>
      <td>Player15</td>
      <td>0.55</td>
      <td>0.43</td>
      <td>0.58</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Away</td>
      <td>PASS</td>
      <td>NaN</td>
      <td>1</td>
      <td>45</td>
      <td>1.80</td>
      <td>61</td>
      <td>2.44</td>
      <td>Player15</td>
      <td>Player19</td>
      <td>0.55</td>
      <td>0.19</td>
      <td>0.45</td>
      <td>0.31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Away</td>
      <td>PASS</td>
      <td>NaN</td>
      <td>1</td>
      <td>77</td>
      <td>3.08</td>
      <td>96</td>
      <td>3.84</td>
      <td>Player19</td>
      <td>Player21</td>
      <td>0.45</td>
      <td>0.32</td>
      <td>0.49</td>
      <td>0.47</td>
    </tr>
  </tbody>
</table>
</div>




```python
events = mio.read_event_data('/Users/lucaspecina/Desktop/Data/Proyectos/1- futbol/Tracking data - Metrica/data',2)
events.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Type</th>
      <th>Subtype</th>
      <th>Period</th>
      <th>Start Frame</th>
      <th>Start Time [s]</th>
      <th>End Frame</th>
      <th>End Time [s]</th>
      <th>From</th>
      <th>To</th>
      <th>Start X</th>
      <th>Start Y</th>
      <th>End X</th>
      <th>End Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Away</td>
      <td>SET PIECE</td>
      <td>KICK OFF</td>
      <td>1</td>
      <td>51</td>
      <td>2.04</td>
      <td>51</td>
      <td>2.04</td>
      <td>Player23</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Away</td>
      <td>PASS</td>
      <td>NaN</td>
      <td>1</td>
      <td>51</td>
      <td>2.04</td>
      <td>87</td>
      <td>3.48</td>
      <td>Player23</td>
      <td>Player20</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.40</td>
      <td>0.51</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Away</td>
      <td>PASS</td>
      <td>NaN</td>
      <td>1</td>
      <td>146</td>
      <td>5.84</td>
      <td>186</td>
      <td>7.44</td>
      <td>Player20</td>
      <td>Player18</td>
      <td>0.43</td>
      <td>0.50</td>
      <td>0.44</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Away</td>
      <td>PASS</td>
      <td>NaN</td>
      <td>1</td>
      <td>248</td>
      <td>9.92</td>
      <td>283</td>
      <td>11.32</td>
      <td>Player18</td>
      <td>Player17</td>
      <td>0.47</td>
      <td>0.19</td>
      <td>0.31</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Away</td>
      <td>PASS</td>
      <td>NaN</td>
      <td>1</td>
      <td>316</td>
      <td>12.64</td>
      <td>346</td>
      <td>13.84</td>
      <td>Player17</td>
      <td>Player16</td>
      <td>0.29</td>
      <td>0.32</td>
      <td>0.26</td>
      <td>0.58</td>
    </tr>
  </tbody>
</table>
</div>




```python
# tracking data

tracking_home= mio.tracking_data('/Users/lucaspecina/Desktop/Data/Proyectos/1- futbol/Tracking data - Metrica/data',2,'Home')
tracking_away= mio.tracking_data('/Users/lucaspecina/Desktop/Data/Proyectos/1- futbol/Tracking data - Metrica/data',2,'Away')
# NICE
tracking_home.head(10)
```

    Reading team: home
    Reading team: away





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Period</th>
      <th>Time [s]</th>
      <th>Home_11_x</th>
      <th>Home_11_y</th>
      <th>Home_1_x</th>
      <th>Home_1_y</th>
      <th>Home_2_x</th>
      <th>Home_2_y</th>
      <th>Home_3_x</th>
      <th>Home_3_y</th>
      <th>...</th>
      <th>Home_10_x</th>
      <th>Home_10_y</th>
      <th>Home_12_x</th>
      <th>Home_12_y</th>
      <th>Home_13_x</th>
      <th>Home_13_y</th>
      <th>Home_14_x</th>
      <th>Home_14_y</th>
      <th>ball_x</th>
      <th>ball_y</th>
    </tr>
    <tr>
      <th>Frame</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.04</td>
      <td>0.94275</td>
      <td>0.50413</td>
      <td>0.64812</td>
      <td>0.28605</td>
      <td>0.67752</td>
      <td>0.42803</td>
      <td>0.69882</td>
      <td>0.55606</td>
      <td>...</td>
      <td>0.50194</td>
      <td>0.61123</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.08</td>
      <td>0.94275</td>
      <td>0.50413</td>
      <td>0.64812</td>
      <td>0.28605</td>
      <td>0.67752</td>
      <td>0.42803</td>
      <td>0.69882</td>
      <td>0.55606</td>
      <td>...</td>
      <td>0.50194</td>
      <td>0.61123</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.12</td>
      <td>0.94275</td>
      <td>0.50413</td>
      <td>0.64812</td>
      <td>0.28605</td>
      <td>0.67752</td>
      <td>0.42803</td>
      <td>0.69882</td>
      <td>0.55606</td>
      <td>...</td>
      <td>0.50194</td>
      <td>0.61123</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.16</td>
      <td>0.94275</td>
      <td>0.50413</td>
      <td>0.64812</td>
      <td>0.28605</td>
      <td>0.67752</td>
      <td>0.42803</td>
      <td>0.69882</td>
      <td>0.55606</td>
      <td>...</td>
      <td>0.50194</td>
      <td>0.61123</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0.20</td>
      <td>0.94275</td>
      <td>0.50413</td>
      <td>0.64812</td>
      <td>0.28605</td>
      <td>0.67752</td>
      <td>0.42803</td>
      <td>0.69882</td>
      <td>0.55606</td>
      <td>...</td>
      <td>0.50194</td>
      <td>0.61123</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0.24</td>
      <td>0.94275</td>
      <td>0.50413</td>
      <td>0.64812</td>
      <td>0.28605</td>
      <td>0.67752</td>
      <td>0.42803</td>
      <td>0.69882</td>
      <td>0.55606</td>
      <td>...</td>
      <td>0.50194</td>
      <td>0.61123</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0.28</td>
      <td>0.94275</td>
      <td>0.50413</td>
      <td>0.64812</td>
      <td>0.28605</td>
      <td>0.67752</td>
      <td>0.42803</td>
      <td>0.69882</td>
      <td>0.55606</td>
      <td>...</td>
      <td>0.50090</td>
      <td>0.64090</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0.32</td>
      <td>0.94275</td>
      <td>0.50413</td>
      <td>0.64812</td>
      <td>0.28605</td>
      <td>0.67752</td>
      <td>0.42803</td>
      <td>0.69882</td>
      <td>0.55606</td>
      <td>...</td>
      <td>0.50073</td>
      <td>0.64646</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0.36</td>
      <td>0.94275</td>
      <td>0.50413</td>
      <td>0.64812</td>
      <td>0.28605</td>
      <td>0.67752</td>
      <td>0.42803</td>
      <td>0.69882</td>
      <td>0.55606</td>
      <td>...</td>
      <td>0.50061</td>
      <td>0.64887</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>0.40</td>
      <td>0.94275</td>
      <td>0.50413</td>
      <td>0.64782</td>
      <td>0.28621</td>
      <td>0.67752</td>
      <td>0.42803</td>
      <td>0.69842</td>
      <td>0.55675</td>
      <td>...</td>
      <td>0.50051</td>
      <td>0.65019</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 32 columns</p>
</div>




```python
# Convert positions from metrica units to meters (note change in Metrica's coordinate system since the last lesson)
tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)
events = mio.to_metric_coordinates(events)
```


```python
# reverse direction of play in the second half so that home team is always attacking from right->left
tracking_home,tracking_away,events = mio.to_single_playing_direction(tracking_home,tracking_away,events)
```


```python
# HACER VIDEO DEL SEGUNDO GOL DEL LOCAL
'''Le seleccionamos los frames que queremos visualizar. Selecciona las imagenes, las junta y plotea'''


PLOTDIR='/Users/lucaspecina/Desktop/Data/Proyectos/1- futbol/Tracking data - Metrica/data'

mviz.save_match_clip(tracking_home.iloc[73600:73600+500],tracking_away.iloc[73600:73600+500],\
                     PLOTDIR,fname='home_goal_2',include_player_velocities=False)
```

    Generating movie...done



```python

```


```python
# Calculate player velocities
tracking_home = mvel.calc_player_velocities(tracking_home,smoothing=True)
tracking_away = mvel.calc_player_velocities(tracking_away,smoothing=True)
```

    /Users/lucaspecina/Desktop/Data/Proyectos/1- futbol/LaurieOnTracking-master/Metrica_Velocities.py:56: RuntimeWarning:
    
    invalid value encountered in greater
    
    /Users/lucaspecina/Desktop/Data/Proyectos/1- futbol/LaurieOnTracking-master/Metrica_Velocities.py:57: RuntimeWarning:
    
    invalid value encountered in greater
    



```python
tracking_away.columns
```




    Index(['Period', 'Time [s]', 'Away_25_x', 'Away_25_y', 'Away_15_x',
           'Away_15_y', 'Away_16_x', 'Away_16_y', 'Away_17_x', 'Away_17_y',
           'Away_18_x', 'Away_18_y', 'Away_19_x', 'Away_19_y', 'Away_20_x',
           'Away_20_y', 'Away_21_x', 'Away_21_y', 'Away_22_x', 'Away_22_y',
           'Away_23_x', 'Away_23_y', 'Away_24_x', 'Away_24_y', 'Away_26_x',
           'Away_26_y', 'ball_x', 'ball_y', 'Away_15_vx', 'Away_15_vy',
           'Away_15_speed', 'Away_16_vx', 'Away_16_vy', 'Away_16_speed',
           'Away_17_vx', 'Away_17_vy', 'Away_17_speed', 'Away_18_vx', 'Away_18_vy',
           'Away_18_speed', 'Away_19_vx', 'Away_19_vy', 'Away_19_speed',
           'Away_20_vx', 'Away_20_vy', 'Away_20_speed', 'Away_21_vx', 'Away_21_vy',
           'Away_21_speed', 'Away_22_vx', 'Away_22_vy', 'Away_22_speed',
           'Away_23_vx', 'Away_23_vy', 'Away_23_speed', 'Away_24_vx', 'Away_24_vy',
           'Away_24_speed', 'Away_25_vx', 'Away_25_vy', 'Away_25_speed',
           'Away_26_vx', 'Away_26_vy', 'Away_26_speed'],
          dtype='object')




```python
# plot a random frame, plotting the player velocities using quivers
mviz.plot_frame( tracking_home.loc[10000], tracking_away.loc[10000],\
                include_player_velocities=True, annotate=True)

```




    (<Figure size 864x576 with 1 Axes>,
     <matplotlib.axes._subplots.AxesSubplot at 0x11698b198>)




![png](output_64_1.png)



```python
# Create a Physical summary dataframe for home players

home_players = np.unique( [ c.split('_')[1] for c in tracking_home.columns if c[:4] == 'Home' ] )
home_summary = pd.DataFrame(index=home_players)

# Calculate minutes played for each player
minutes = []
for player in home_players:
    # search for first and last frames that we have a position observation for each player (when a player is not on the pitch positions are NaN)
    column = 'Home_' + player + '_x' # use player x-position coordinate
    player_minutes = ( tracking_home[column].last_valid_index() - tracking_home[column].first_valid_index() + 1 ) / 25 / 60. # convert to minutes
    minutes.append( player_minutes )
home_summary['Minutes Played'] = minutes
home_summary = home_summary.sort_values(['Minutes Played'], ascending=False)
home_summary.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Minutes Played</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>94.104</td>
    </tr>
    <tr>
      <th>11</th>
      <td>94.104</td>
    </tr>
    <tr>
      <th>2</th>
      <td>94.104</td>
    </tr>
    <tr>
      <th>3</th>
      <td>94.104</td>
    </tr>
    <tr>
      <th>5</th>
      <td>94.104</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Calculate total distance covered for each player
distance = []
for player in home_summary.index:
    column = 'Home_' + player + '_speed'
    player_distance = tracking_home[column].sum()/25./1000 # this is the sum of the distance travelled from one observation to the next (1/25 = 40ms) in km.
    distance.append( player_distance )
home_summary['Distance [km]'] = distance
home_summary.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Minutes Played</th>
      <th>Distance [km]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>94.104</td>
      <td>10.369966</td>
    </tr>
    <tr>
      <th>11</th>
      <td>94.104</td>
      <td>5.203820</td>
    </tr>
    <tr>
      <th>2</th>
      <td>94.104</td>
      <td>9.845300</td>
    </tr>
    <tr>
      <th>3</th>
      <td>94.104</td>
      <td>9.546312</td>
    </tr>
    <tr>
      <th>5</th>
      <td>94.104</td>
      <td>11.909182</td>
    </tr>
  </tbody>
</table>
</div>




```python
# make a simple bar chart of distance covered for each player
plt.subplots()
ax = home_summary['Distance [km]'].sort_values().plot.bar(rot=0)
ax.set_xlabel('Player')
ax.set_ylabel('Distance covered [km]')
```




    Text(0, 0.5, 'Distance covered [km]')




![png](output_67_1.png)



```python
# plot positions at KO (to find out what position each player is playing)
mviz.plot_frame( tracking_home.loc[51], tracking_away.loc[51], include_player_velocities=False, annotate=True)
```




    (<Figure size 864x576 with 1 Axes>,
     <matplotlib.axes._subplots.AxesSubplot at 0x130bac3c8>)




![png](output_68_1.png)



```python
# now calculate distance covered while: walking, joggings, running, sprinting
walking = []
jogging = []
running = []
sprinting = []
for player in home_summary.index:
    column = 'Home_' + player + '_speed'
    # walking (less than 2 m/s)
    player_distance = tracking_home.loc[tracking_home[column] < 2, column].sum()/25./1000
    walking.append( player_distance )
    # jogging (between 2 and 4 m/s)
    player_distance = tracking_home.loc[ (tracking_home[column] >= 2) & (tracking_home[column] < 4), column].sum()/25./1000
    jogging.append( player_distance )
    # running (between 4 and 7 m/s)
    player_distance = tracking_home.loc[ (tracking_home[column] >= 4) & (tracking_home[column] < 7), column].sum()/25./1000
    running.append( player_distance )
    # sprinting (greater than 7 m/s)
    player_distance = tracking_home.loc[ tracking_home[column] >= 7, column].sum()/25./1000
    sprinting.append( player_distance )
    
home_summary['Walking [km]'] = walking
home_summary['Jogging [km]'] = jogging
home_summary['Running [km]'] = running
home_summary['Sprinting [km]'] = sprinting

# make a clustered bar chart of distance covered for each player at each speed
ax = home_summary[['Walking [km]','Jogging [km]','Running [km]','Sprinting [km]']].plot.bar(colormap='coolwarm')
ax.set_xlabel('Player')
ax.set_ylabel('Distance covered [m]')
```




    Text(0, 0.5, 'Distance covered [m]')




![png](output_69_1.png)



```python
'''Los sprints de mas de un segundo'''
# sustained sprints: how many sustained sprints per match did each player complete? Defined as maintaining a speed > 7 m/s for at least 1 second
nsprints = []
sprint_threshold = 7 # minimum speed to be defined as a sprint (m/s)
sprint_window = 1*25 # minimum duration sprint should be sustained (in this case, 1 second = 25 consecutive frames)
for player in home_summary.index:
    column = 'Home_' + player + '_speed'
    # trick here is to convolve speed with a window of size 'sprint_window', and find number of occassions that sprint was sustained for at least one window length
    # diff helps us to identify when the window starts
    player_sprints = np.diff( 1*( np.convolve( 1*(tracking_home[column]>=sprint_threshold), np.ones(sprint_window), mode='same' ) >= sprint_window ) )
    nsprints.append( np.sum( player_sprints == 1 ) )
home_summary['# sprints'] = nsprints
home_summary['# sprints'].sort_values(ascending=False)
```




    10    13
    7      9
    5      8
    3      8
    9      7
    6      7
    8      6
    1      5
    2      3
    13     2
    12     2
    14     1
    4      0
    11     0
    Name: # sprints, dtype: int64




```python
'''Donde fueron los sprints del jugador 10. FIJARSE COMO FUNCIONA EL PLOT'''
# Plot the trajectories for each of player 10's sprints
player = '10'
column = 'Home_' + player + '_speed' # spped
column_x = 'Home_' + player + '_x' # x position
column_y = 'Home_' + player + '_y' # y position
# same trick as before to find start and end indices of windows of size 'sprint_window' in which player speed was above the sprint_threshold
player_sprints = np.diff( 1*( np.convolve( 1*(tracking_home[column]>=sprint_threshold), np.ones(sprint_window), mode='same' ) >= sprint_window ) )
player_sprints_start = np.where( player_sprints == 1 )[0] - int(sprint_window/2) + 1 # adding sprint_window/2 because of the way that the convolution is centred
player_sprints_end = np.where( player_sprints == -1 )[0] + int(sprint_window/2) + 1

# now plot all the sprints
fig,ax = mviz.plot_pitch()
for s,e in zip(player_sprints_start,player_sprints_end):
    ax.plot(tracking_home[column_x].iloc[s],tracking_home[column_y].iloc[s],'ro')
    ax.plot(tracking_home[column_x].iloc[s:e+1],tracking_home[column_y].iloc[s:e+1],'r')
```


![png](output_71_0.png)



```python
events.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Type</th>
      <th>Subtype</th>
      <th>Period</th>
      <th>Start Frame</th>
      <th>Start Time [s]</th>
      <th>End Frame</th>
      <th>End Time [s]</th>
      <th>From</th>
      <th>To</th>
      <th>Start X</th>
      <th>Start Y</th>
      <th>End X</th>
      <th>End Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Away</td>
      <td>SET PIECE</td>
      <td>KICK OFF</td>
      <td>1</td>
      <td>51</td>
      <td>2.04</td>
      <td>51</td>
      <td>2.04</td>
      <td>Player23</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Away</td>
      <td>PASS</td>
      <td>NaN</td>
      <td>1</td>
      <td>51</td>
      <td>2.04</td>
      <td>87</td>
      <td>3.48</td>
      <td>Player23</td>
      <td>Player20</td>
      <td>0.00</td>
      <td>-0.00</td>
      <td>-10.60</td>
      <td>-0.68</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Away</td>
      <td>PASS</td>
      <td>NaN</td>
      <td>1</td>
      <td>146</td>
      <td>5.84</td>
      <td>186</td>
      <td>7.44</td>
      <td>Player20</td>
      <td>Player18</td>
      <td>-7.42</td>
      <td>-0.00</td>
      <td>-6.36</td>
      <td>19.04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Away</td>
      <td>PASS</td>
      <td>NaN</td>
      <td>1</td>
      <td>248</td>
      <td>9.92</td>
      <td>283</td>
      <td>11.32</td>
      <td>Player18</td>
      <td>Player17</td>
      <td>-3.18</td>
      <td>21.08</td>
      <td>-20.14</td>
      <td>14.96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Away</td>
      <td>PASS</td>
      <td>NaN</td>
      <td>1</td>
      <td>316</td>
      <td>12.64</td>
      <td>346</td>
      <td>13.84</td>
      <td>Player17</td>
      <td>Player16</td>
      <td>-22.26</td>
      <td>12.24</td>
      <td>-25.44</td>
      <td>-5.44</td>
    </tr>
  </tbody>
</table>
</div>




```python
# HACER VIDEOS
'''Le seleccionamos los frames que queremos visualizar. Selecciona las imagenes, las junta y plotea'''

PLOTDIR='/Users/lucaspecina/Desktop/Data/Proyectos/1- futbol/Tracking data - Metrica/data'

frame_video = 83000
cantidad = 5000

# construir el video
mviz.save_match_clip(tracking_home.iloc[frame_video:frame_video+cantidad],tracking_away.iloc\
                     [frame_video:frame_video+cantidad],PLOTDIR,fname='test',include_player_velocities=False)

# leerlo y reproducirlo
from IPython.display import Video
Video("/Users/lucaspecina/Desktop/Data/Proyectos/1- futbol/Tracking data - Metrica/data/home_goal_2.mp4",\
      embed=True,width=900,height=500)
```

    Generating movie...done



```python

```




<video controls  width="900"  height="500">
 <source src="data:video/mp4;base64,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AAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAIAAAAAAIAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAIAAAAAAIAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAGAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACAAAAAACAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAgAAAAAAgAAAgAAAAABAAAEAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAABgAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAABAAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAgAAAAAAgAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAIAAAAAAIAAAIAAAAAHHN0c2MAAAAAAAAAAQAAAAEAAAH0AAAAAQAAB+RzdHN6AAAAAAAAAAAAAAH0AAAuDwAABKUAAAI0AAABRAAAAUkAAASJAAAB7AAAAWYAAAGVAAAEnAAAAmoAAAEoAAABYQAABC0AAAHsAAABhgAAAWsAAAQcAAAB3QAAAVcAAAFDAAAFhQAAAmgAAAGPAAABXQAABC4AAAIJAAABjQAAAWoAAARzAAACcgAAAV8AAAFKAAAEcwAAAfYAAAF1AAABkgAABMUAAAI7AAABaQAAAXQAAAUAAAACZwAAAW8AAAG+AAAE0QAAAmwAAAHPAAAByQAAA8sAAAF7AAABRgAAA/oAAAHkAAABUgAAAU8AAAP5AAACHAAAAVEAAAFeAAADpgAAAdsAAAFoAAABVwAABJsAAAKyAAABggAAAaQAAARSAAACOwAAAYwAAAHOAAAFFgAAAigAAAG1AAABiAAABKIAAAIrAAABiAAAAV8AAAPVAAAB+QAAAWgAAAGaAAAE0QAAAlUAAAE+AAABfwAABFcAAAH0AAABiQAAAbQAAATBAAACegAAAbsAAAG3AAAEkAAAAe4AAAGxAAAFcgAAAnQAAAF4AAABgwAABMgAAAHeAAABcwAABIEAAAG+AAABYAAAATwAAANoAAABKwAAA8wAAAG5AAABPAAAAQIAAARDAAACGwAAATcAAAF9AAAEdAAAAjMAAAFoAAABdwAABHcAAAGbAAABdgAABMkAAAI9AAABYgAAATsAAASCAAAB9AAAAVYAAAEbAAAD+QAAAWgAAAFRAAABUAAAA/wAAAJSAAABJwAAAWwAAAReAAAB+QAAAU4AAAF3AAAFQwAAAi0AAAGIAAABcAAABAYAAAI9AAABUQAAASoAAAN2AAAB5gAAAVYAAAFQAAAEMgAAAfYAAAE+AAABQAAABCMAAAG9AAABLQAAAU0AAAP8AAAB6wAAATwAAAEjAAADxQAAAagAAAEqAAABHAAAA+wAAAGtAAABJAAAAVEAAARoAAAB1wAAAPwAAAE2AAAD6AAAAa0AAAEEAAABKAAABGAAAAIHAAABdAAAAU8AAAPwAAACOQAAAVoAAAFkAAAENQAAAfMAAAFVAAABawAABJ8AAAHXAAABWAAAAR4AAATxAAABzgAAAUEAAAE0AAAEzgAAAjgAAAF8AAABQAAABR8AAAKQAAABbwAAAY0AAAWgAAACKgAAAYAAAAFhAAAFbQAAAokAAAFdAAABdAAABV0AAAJMAAABQwAAAYcAAARTAAACSQAAAWsAAAETAAAEvwAAAhgAAAEtAAABBwAAA/EAAAHkAAABEAAAATgAAARTAAACkgAAAWkAAAFxAAAExwAAAeUAAAFwAAABPgAAAi0AAAGCAAABOQAAM9EAAARuAAABqAAAASsAAAEEAAAD0AAAAWIAAAEIAAABGgAABFsAAAHvAAAA3AAAARUAAARDAAABdgAAAOoAAAEKAAAEZgAAAesAAAFiAAABKgAABPQAAAHjAAABEwAAASEAAATtAAABsAAAARcAAAEsAAAEtgAAAg0AAAD8AAABGwAABTEAAAGwAAABDwAAATAAAATrAAAB1AAAARkAAAESAAAFRAAAAfwAAAENAAAA9gAABH8AAAHNAAABIwAAAScAAAVYAAABsgAAAL0AAAEGAAAEvwAAAdIAAAEbAAABGwAABRcAAAHRAAABRAAAARwAAATEAAAB5wAAAPgAAADpAAADvAAAAO4AAARCAAABaAAAARIAAADnAAAFLQAAAfIAAAEyAAABCAAABI8AAAFaAAABCAAAAScAAAO5AAABtAAAANQAAAD2AAAEZAAAAWsAAADfAAABEAAABOMAAAHHAAABEAAAARUAAAPFAAAB6gAAAM8AAADzAAAETgAAAasAAAD6AAABDQAABTgAAAI0AAABVQAAAQEAAASWAAABugAAAPMAAAFCAAAEgQAAAisAAAD1AAAAxwAABDkAAAIaAAAA/QAAANkAAARrAAAB7QAAAOAAAAEkAAACRwAABAMAAAGQAAABMAAAAQYAAAVRAAABTAAAARMAAATgAAABygAAARwAAADkAAAExQAAAc0AAAEFAAABBAAABNcAAAHGAAABFwAAATUAAARMAAACHgAAAQkAAAEbAAAEHQAAAZ0AAADxAAAA/wAABbUAAAHwAAABDwAAAOwAAAVIAAAB8QAAAOoAAADaAAAEaQAAAXIAAADbAAAA8wAABH0AAAHxAAAAmgAAAPkAAAPaAAABTgAAAM4AAADfAAAEbgAAAYYAAAEBAAAAsgAABHkAAAFoAAAA4AAAAL4AAAQnAAABWAAAAOoAAADxAAAEOwAAAaoAAADqAAAA4QAABIYAAAFtAAABCwAAAQQAAARoAAABxAAAARYAAAD9AAAEQgAAAeUAAAD0AAABAQAABAUAAAGOAAABGQAAASEAAAThAAABvgAAAPIAAADpAAAD0gAAAUYAAADeAAAA4AAAA30AAAGLAAABDgAAAO8AAANmAAABpQAAAOYAAADgAAADvgAAAZYAAAEDAAABGQAAA+oAAAHkAAAA6wAAAS8AAAQEAAABzgAAARkAAAElAAAEIAAAAboAAAD5AAABBQAABJAAAAJNAAABHAAAAPYAAAP3AAAB2gAAARMAAAE2AAAEaQAAAkoAAAEzAAABNQAAAzQAAAIYAAABNgAAAWcAAAH7AAABkAAAARwAAAAUc3RjbwAAAAAAAAABAAAAMAAAANt1ZHRhAAAA021ldGEAAAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAApmlsc3QAAAAlqW5hbQAAAB1kYXRhAAAAAQAAAABUcmFja2luZyBEYXRhAAAAIqlBUlQAAAAaZGF0YQAAAAEAAAAATWF0cGxvdGxpYgAAACWpdG9vAAAAHWRhdGEAAAABAAAAAExhdmY1OC4yOS4xMDAAAAAyqWNtdAAAACpkYXRhAAAAAQAAAABNZXRyaWNhIHRyYWNraW5nIGRhdGEgY2xpcA==" type="video/mp4">
 Your browser does not support the video tag.
 </video>




```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

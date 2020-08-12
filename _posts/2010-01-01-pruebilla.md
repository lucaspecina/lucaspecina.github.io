```python
# import plotly.express as px
# import plotly.offline as py_offline
# import plotly.graph_objs as go
# py_offline.init_notebook_mode()
```


```python
df = px.data.iris()

fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                 size='petal_length', hover_data=['petal_width'])
py.iplot(fig,filename='pruebaloca')
```





<iframe
    width="100%"
    height="600px"
    src="https://plotly.com/~lucaspecina/3.embed"
    frameborder="0"
    allowfullscreen
></iframe>





```python

```


```python

```


```python

```


```python

```

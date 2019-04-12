#https://assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf
#cheat sheet
import pandas as pd

s = pd.Series([3, -5, 7, 4], index=['a', 'b', 'c', 'd'])

print(s)

data = {'Country': ['Belgium', 'India', 'Brazil'],
        'Capital': ['Brussels', 'New Delhi', 'Brasília'],
        'Population': [11190846, 1303171035, 207847528]}

df = pd.DataFrame(data, columns=['Country', 'Capital', 'Population'])

print(df)

print(s['b'])
print(df[1:])
#print(df.iloc([0],[0]))
#print(df.iat([0],[0]))



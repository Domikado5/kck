import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.legend_handler import HandlerLine2D

files = [
    {'label': '1-Evol-RS', 'name': 'rsel.csv', 'color': 'blue', 'marker': 'o'},
    {'label': '1-Coev-RS', 'name': 'cel-rs.csv', 'color': 'green', 'marker': 'v'},
    {'label': '2-Coev-RS', 'name': '2cel-rs.csv', 'color': 'red', 'marker': 'D'},
    {'label': '1-Coev', 'name': 'cel.csv', 'color': 'black', 'marker': 's'},
    {'label': '2-Coev', 'name': '2cel.csv', 'color': 'purple', 'marker': 'd'},
]

fig, (ax1, ax2) = plt.subplots(1, 2)
ax11 = ax1.twiny()
for file in files:
    data = pd.read_csv(file['name'])
    data['avg'] = data.iloc[:, 2:].copy().mean(axis=1)
    # data['last_avg'] = data.iloc[-1, 2:]
    ax1.plot(
        data['effort']/1000, data['avg']*100, 
        label=file['label'], color=file['color'], 
        linestyle='solid', linewidth=1,
        marker=file['marker'], markevery=25,
        markersize=5, markeredgecolor='black',
        markeredgewidth=0.5)
    ax2.boxplot(
        data['avg']*100
    )
ax1.axis(xmin=0, xmax=500, ymin=60, ymax=100)
ax1.set_xlabel("Rozegranych gier (x1000)")
ax1.set_ylabel("Odsetek wygranych gier [%]")
ax1.legend([file['label'] for file in files], numpoints=2)
ax1.grid(color='gray', linestyle='dotted', dashes=(1,4))
ax1.tick_params(direction='in')
ax11.set_xlim(xmin=0, xmax=200)
ax11.set_xticks([0, 40, 80, 120, 160, 200])
ax11.set_xlabel('Pokolenie')
ax11.tick_params(direction='in')
plt.savefig('wykres.pdf')
plt.close()
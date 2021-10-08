import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import font_manager

files = [
    {'label': '1-Evol-RS', 'name': 'rsel.csv', 'color': 'blue', 'marker': 'o'},
    {'label': '1-Coev-RS', 'name': 'cel-rs.csv', 'color': 'green', 'marker': 'v'},
    {'label': '2-Coev-RS', 'name': '2cel-rs.csv', 'color': 'red', 'marker': 'D'},
    {'label': '1-Coev', 'name': 'cel.csv', 'color': 'black', 'marker': 's'},
    {'label': '2-Coev', 'name': '2cel.csv', 'color': 'purple', 'marker': 'd'},
]
# Display available fonts
# for font in sorted([font.name for font in font_manager.fontManager.ttflist]):
    # print(font)

plt.rc('font', family='CMU Serif', size=9) 

fig, (ax1, ax2) = plt.subplots(1, 2)
box_values = []
ax11 = ax1.twiny()
ax12 = ax1.twinx()
ax21 = ax2.twiny()

for file in files:
    data = pd.read_csv(file['name'])
    box_values.append((np.array(data.iloc[-1:, 2:])*100)[0].tolist())
    data['avg'] = data.iloc[:, 2:].copy().mean(axis=1)
    ax1.plot(
        data['effort']/1000, data['avg']*100, 
        label=file['label'], color=file['color'], 
        linestyle='solid', linewidth=1,
        marker=file['marker'], markevery=25,
        markersize=5, markeredgecolor='black',
        markeredgewidth=0.5)

# Line plot
ax1.axis(xmin=0, xmax=500, ymin=60, ymax=100)
ax1.set_xlabel("Rozegranych gier (x1000)")
ax1.set_ylabel("Odsetek wygranych gier [%]")
ax1.legend([file['label'] for file in files], numpoints=2)
ax1.grid(color='gray', linestyle='-', dashes=(2,6), linewidth=0.45)
ax1.tick_params(direction='in')

#Secondary X axis
ax11.set_xlim(xmin=0, xmax=200)
ax11.set_xticks([0, 40, 80, 120, 160, 200])
ax11.set_xlabel('Pokolenie')
ax11.tick_params(direction='in')

#Secondary Y axis
ax12.yaxis.tick_right()
ax12.set_ylim(ax1.get_ylim())
ax12.set_yticklabels([])
ax12.tick_params(direction='in')

#Box plot
ax2.boxplot(
    box_values, notch=True, 
    showmeans=True,
    flierprops={'marker': '+', 'markeredgecolor': 'blue', 'markersize': 6, 'markeredgewidth': 0.5},
    boxprops={'color': 'blue'},
    medianprops={'color': 'red'},
    meanprops={'marker': 'o', 'markerfacecolor': 'blue', 'markeredgecolor': 'black', 'markersize': 4.5},
    whiskerprops={'linestyle': 'dashed', 'color': 'blue', 'dashes': (6,5)}
)
ax2.grid(color='gray', linestyle='-', dashes=(2,6), linewidth=0.45)
ax2.tick_params(direction='in')
ax2.set_xticklabels([file['label'] for file in files], rotation=20)
ax2.set_ylim(ymin=60, ymax=100)
ax2.yaxis.tick_right()

#Secondary X axis
ax21.set_xlim(ax2.get_xlim())
ax21.set_xticklabels([])
ax21.tick_params(direction='in')
plt.savefig('wykres.pdf')
plt.close()
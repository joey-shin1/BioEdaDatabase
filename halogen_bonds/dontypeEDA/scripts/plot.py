import matplotlib.pyplot as plt
import pandas as pd

def eda_pes(ax, df, fontsize=None):
    items = ['ELEC', 'MOD_PAULI', 'POL', 'DISP', 'CT', 'INTERACTION']
    labels = ['ELEC', 'MOD_PAULI', 'POL', 'DISP', 'CT', 'INT']
    for item, label in zip(items, labels):
        ax.scatter(x=df['Dist'], y=df[item], label=label)
    ax.legend(fontsize=fontsize)
    return ax

def eda_pes_plot(ax, df):
    # items = ['FRZ', 'POL', 'DISP', 'CT', 'INTERACTION']
    # labels = ['FRZ', 'POL', 'DISP', 'CT', 'INT']
    items = ['ELEC', 'MOD_PAULI', 'POL', 'DISP', 'CT', 'INTERACTION']
    labels = ['ELEC', 'MOD_PAULI', 'POL', 'DISP', 'CT', 'INT']
    for item, label in zip(items, labels):
        ax.plot(df['Dist'], df[item], label=label)
    return ax

def calc_portion(series, ratio=False):
    items = ['FRZ', 'POL', 'DISP', 'CT', 'INTERACTION']
    portion = {}

    if ratio:
        for item in items:
            data = series[item] / series['INTERACTION']
            portion[item] = data
    else: 
        for item in items:
            data = series[item]
            portion[item] = data
    return portion

def energy_bar_ax(ax, portions, xlabels, items, ratio=False):    
    xlocs = {}
    yvals = {}
    
    start = 1
    bar_width = 1
    pad_width = 2
    
    centers = [start + i * (len(items) * bar_width + pad_width) for i in range(len(portions))]
    for i, item in enumerate(items):
        xlocs[item] = [center - bar_width * (len(items) - 1) / 2 + i * bar_width for center in centers]
        yvals[item] = [p[item] for p in portions]
    
    for item in items:
        ax.bar(xlocs[item], yvals[item], label=item, width=bar_width - 0.1)
    ax.legend()
    ax.set_xticks(centers)
    ax.set_xticklabels(xlabels)
    if ratio:
        ax.set_ylabel('Ratio', fontsize=16)
    else:
        ax.set_ylabel('Energy (kJ/mol)', fontsize=16)

    return ax
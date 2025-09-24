import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.axes import Axes
import matplotlib.patches as patches
from matplotlib.patches import Patch
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem, Draw


def plot(type, args, dfs, ax, cmap_range=[-100, 0.0], median=False, ratio1=False, ratio2=False, plot=False, kde_norm=False):
    """
    General Plot Function

    Parameters
    ----------
    type: {'ternary', 'bar', 'dist', 'box'}
        Plot types
    args: list
        List of arguments for each plot type:
        - 'ternary': (label, index, type, cmap_range)
            - label: str
                type of atom/interaction
            - index: str
                index for ternary plot
            - type: str
                type of ternary plot ('frz_orb_disp', 'elec_orb_disp', 'frz_pol_ct')
        - 'bar': (labels, items)
            - label: list(str)
                type of atom/interaction
            - items: list(str)
                EDA energy components 
        - 'dist': (column, labels, item, palette)
            - column: str
                column of dataframe
            - label: list(str)
                type of atom/interaction
            - item: list(str)
                EDA energy components
            - palette: sns.color_palette
                seaborn color palette
        - 'box': (dists, xlabels, items)
            - dists: list(float)
                list of distances from lowest to highest by intervals
            - xlabels: list(str)
                list of distances for labeling on x axis
            - items: list(str)
                EDA energy components 
    df: pd.DataFrame or list of pd.DataFrame
        Interaction Dataframe with EDA energies
    ax: ax
        ax
    cmap_range: [int, int]
        color map energy range for ternary plot from lowest to highest
    median: bool
        plot median for bar plot
    ratio1: bool
        ratio1 for bar plot
    ratio2: bool
        ratio2 for bar plot
    plot: bool
        plot lines for box plot
    kde_norm: bool
        normalize highest value of each distributions to one in dist plot
    Returns
    -------
    sc: ax
        plot
    """
    if type == 'bar': #temp replace DISP with CLS_DISP for no_Orth_Decomposition
        for df in dfs:
            df.loc[df['DISP'].isna(), 'DISP'] = df['CLS_DISP']
    else:
        dfs.loc[dfs['DISP'].isna(), 'DISP'] = dfs['CLS_DISP']
    
    if type == 'ternary':
        label, index, type = args
        return ternary_ax(dfs, ax, title=f'({index}) {dfs.shape[0]} {label}', type=type, cmap_range=cmap_range)
    elif type == 'bar':
        labels, items = args
        ratio = ratio1 or ratio2
        portions = [calc_portion(df, median=median, ratio1=ratio1, ratio2=ratio2) for df in dfs]
        return energy_bar_ax(ax, portions, labels, items, ratio=ratio)
    elif type == 'dist':
        column, labels, item, palette = args
        if kde_norm:
            kdeplot_norm(ax, dfs, column, labels, item, palette)
        else:
            kdeplot(ax, dfs, column, labels, item, palette)
    elif type == 'box':
        dists, xlabels, items = args
        data = box_plot_data(dfs, dists)
        return grouped_boxplot_ax(ax, data, xlabels, items, plot=plot)
    else:
        return None
        

def plot_mols(df):
    mols = []
    for smi1, smi2 in zip(df['smiles0'], df['smiles1']):
        m = Chem.MolFromSmiles(f"{smi1}.{smi2}")
        AllChem.Compute2DCoords(m)
        mols.append(m)

    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 300))
    return img


def kdeplot(ax, df, column, labels, item, palette):
    for index, label in enumerate(labels):
        subset = df[df[column] == label]
        sns.kdeplot(subset[item], bw_adjust=0.5, label=label, ax=ax, color=palette[index])
    ax.legend(title=f'{column} type')
    ax.set_title(f'{item} Energy')
    ax.set_ylabel("Density")
    ax.set_xlabel(f"{item}")


def kdeplot_norm(ax, df, column, labels, item, palette):
    for index, label in enumerate(labels):
        subset = df[df[column] == label]    
        line = sns.kdeplot(subset[item], bw_adjust=0.5, ax=ax)
        x, y = line.get_lines()[-1].get_data()
        line.get_lines()[-1].remove()
        y /= y.max()
        ax.plot(x, y, label=label, color=palette[index])
    ax.legend(title=f'{column} type')
    ax.set_title(f'{item} Energy Normalized')
    ax.set_ylabel("Normalized Density")
    ax.set_xlabel(f"{item}")

    
def grouped_boxplot_ax(ax, data, xlabels, items, plot=False):
    xlocs = {}
    start = 1
    box_width = 0.6
    box_pad = 0.2
    pad_width = 1

    colors = [color["color"] for color in plt.rcParams["axes.prop_cycle"]]
    centers = [start + i * (len(items) * (box_width + box_pad) + pad_width) for i in range(len(data))]

    for i, item in enumerate(items):
        xlocs[item] = [center - ((len(items) - 1) * (box_width + box_pad)) / 2 + i * (box_width + box_pad) for center in centers]

    box_patches = []
    for i, item in enumerate(items):
        box_data = [d[item] for d in data] 
        bp = ax.boxplot(box_data, positions=xlocs[item], widths=box_width, patch_artist=True, showfliers=False, boxprops=dict(facecolor=colors[i]))

        if plot:
            avgs = []
            for d in data:
                avgs.append(d[item].median())
            ax.plot(xlocs[item], avgs)
        
        if bp['boxes']:
            box_patches.append(Patch(facecolor=colors[i], label=item))
            
    ax.set_xticks(centers)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel('Values', fontsize=14)
    ax.grid()
    ax.legend(handles=box_patches)
    
    return ax


def box_plot_data(df, dists):
    data = []
    for index in range(len(dists) - 1):
        data.append(df.query(f'dist > {dists[index]} & dist < {dists[index+1]}'))
    return data


def energy_bar_ax(ax, portions, xlabels, items, ratio=False):    
    xlocs = {}
    yvals = {}
    
    start = 1
    bar_width = 1
    pad_width = 2
    
    centers = [start + i * (len(items) * bar_width + pad_width) for i in range(len(portions))]
    for i, item in enumerate(items):
        xlocs[item] = [center - bar_width * (len(items) - 1) / 2 + i * bar_width for center in centers]
        yvals[item] = [p[item][0] for p in portions]
    
    for item in items:
        ax.bar(xlocs[item], yvals[item], label=item, width=bar_width - 0.1)
    ax.legend()
    ax.set_xticks(centers)
    ax.set_xticklabels(xlabels)
    if ratio:
        ax.set_ylabel('Ratio', fontsize=16, family='Times New Roman')
    else:
        ax.set_ylabel('Energy (kJ/mol)', fontsize=16, family='Times New Roman')

    return ax


def calc_portion(df, median=False, ratio1=False, ratio2=False):
    items = ['FROZEN', 'CLS_ELEC', 'MOD_PAULI', 'DISP', 'PAULI+DISP', 'POLARIZATION', 'CHARGE_TRANSFER', 'TOTAL']
    portion = {}

    def aggregate(data):
        if median:
            return data.median()
        else:
            return data.mean()
    
    for item in items:
        if item == 'PAULI+DISP':
            data = df['MOD_PAULI'] + df['DISP']
        else:
            data = df[item]
        
        if ratio1:
            energy = df['TOTAL']
            portion[item] = (aggregate(data) / aggregate(energy), data.std() / energy.std())
        elif ratio2:
            data = data / df['TOTAL']
            portion[item] = (aggregate(data), data.std())
        else:
            portion[item] = (aggregate(data), data.std())
    return portion


def ternary_ax(eda_df: pd.DataFrame, ax: Axes, title: str = '', type='frz_orb_disp', cmap_range=[-100, 0.0]):
    """
    Create Ternary plot for given data+condition

    Parameters
    ----------
    eda_df: pd.DataFrame
        Interaction Dataframe with EDA energies
    title: str
        Title of Ternary plot
    type: str
        Type of ternary plot
    cmap_range: array of int
        range of energy for cmap 

    Returns
    -------
    sc: ax
        Ternary plot
    """
    ax.set_xlim([-0.75, 1.25])
    ax.set_ylim([-0.18, 1.02])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    # form and color ternary triangles
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    pathPos = Path([(0., 0.), (1., 0.), (0.5, 0.866), (0., 0.)], codes)
    pathNeg = Path([(0., 0.), (-0.5, 0.866), (0.5, 0.866), (0., 0.)], codes)
    ax.add_patch(patches.PathPatch(pathPos, facecolor='white', lw=2))
    ax.add_patch(patches.PathPatch(pathNeg, facecolor='#fff5ee', lw=2))

    # form and color HB/MX/DD dividing lines
    # ax.plot([0.667, 0.5], [0., 0.866], color='#eeb4b4', lw=0.5)
    # ax.plot([-0.333, 0.5], [0.577, 0.866], color='#eeb4b4', lw=0.5)
    # ax.plot([0.333, 0.5], [0., 0.866], color='#7ec0ee', lw=0.5)
    # ax.plot([-0.167, 0.5], [0.289, 0.866], color='#7ec0ee', lw=0.5)

    # label corners
    if type == 'frz_orb_disp':
        tags = ['Elec+Pauli', 'Orb', 'Disp']
        datas = np.vstack(([
            eda_df['CLS_ELEC'] + eda_df['MOD_PAULI'],
            eda_df['CHARGE_TRANSFER'] + eda_df['POLARIZATION'],
            eda_df['DISP'],
            eda_df['TOTAL']
        ])).T
    elif type == 'elec_orb_disp':
        tags = ['Elec', 'Orb', 'Disp']
        datas = np.vstack(([
            eda_df['CLS_ELEC'],
            eda_df['CHARGE_TRANSFER'] + eda_df['POLARIZATION'],
            eda_df['DISP'],
            eda_df['TOTAL']
        ])).T
    elif type == 'frz_pol_ct':
        tags = ['Frz', 'Pol', 'CT']
        datas = np.vstack(([
            eda_df['FROZEN'],
            eda_df['POLARIZATION'],
            eda_df['CHARGE_TRANSFER'],
            eda_df['TOTAL']
        ])).T


    fontsize = 16
    ax.text(1.0, -0.15, tags[0] + ' (\u2212)',
        verticalalignment='bottom', horizontalalignment='center',
        family='Times New Roman', weight='bold', fontsize=fontsize)
    ax.text(0.5, 0.9, tags[1] + u' (\u2212)',
        verticalalignment='bottom', horizontalalignment='center',
        family='Times New Roman', weight='bold', fontsize=fontsize)
    ax.text(0.0, -0.15, tags[2] + u' (\u2212)',
        verticalalignment='bottom', horizontalalignment='center',
        family='Times New Roman', weight='bold', fontsize=fontsize)
    ax.text(-0.5, 0.9, tags[0] + ' (+)',
        verticalalignment='bottom', horizontalalignment='center',
        family='Times New Roman', weight='bold', fontsize=fontsize)

    xvals = []
    yvals = []
    cvals = []

    for data in datas:
        # calc ternary posn and color
        Ftop = abs(data[1]) / (abs(data[0]) + abs(data[1]) + abs(data[2]))
        Fright = abs(data[0]) / (abs(data[0]) + abs(data[1]) + abs(data[2]))
        xdot = 0.5 * Ftop + Fright
        ydot = 0.866 * Ftop
        # cdot = 0.5 + (xdot - 0.5) / (1. - Ftop)
        cdot = data[-1]
        if data[0] > 0.:
            xdot = 0.5 * (Ftop - Fright)
            ydot = 0.866 * (Ftop + Fright)
        # print(elst, orb, disp, '', xdot, ydot, cdot)

        xvals.append(xdot)
        yvals.append(ydot)
        cvals.append(cdot)

    sc = ax.scatter(
        xvals, yvals, c=cvals, 
        s=10.0,
        marker="o",
        cmap=mpl.cm.jet, 
        edgecolor='none', 
        vmin=cmap_range[0], 
        vmax=cmap_range[1], 
        zorder=10
    )

    # remove figure outline
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if title:
        ax.set_title(title, fontsize=18, family='Times New Roman', weight='bold', loc='left')
    
    return sc

def ternary(eda_df, title='', savefile=None, type='frz_orb_disp', plot_cbar=True):
    """
    Make Ternary Plot of EDA output    
    """
    # initialize plot
    fig, ax = plt.subplots(figsize=(6, 3.6))

    sc = ternary_ax(eda_df, ax, title, type)

    if plot_cbar:
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Total Interaction Energy (kJ/mol)')

    # save and show
    if savefile:
        fig.savefig(savefile, transparent=True, bbox_inches='tight',
                    dpi=450, edgecolor='none', pad_inches=0.0)
    # if view:
    #     plt.show()
    # plt.close()
    
    return fig
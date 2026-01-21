import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.axes import Axes
import matplotlib.patches as patches
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import seaborn as sns

from matplotlib import font_manager
font_manager.fontManager.addfont("./edautils/fonts/Lato/Lato-Regular.ttf")
font_manager.fontManager.addfont("./edautils/fonts/Helvetica/Helvetica.ttf")

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

plt.rcParams.update({
    # 'font.size': 12,
    # 'axes.titlesize': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Lato'],
})

COLOR_MAPS = {
    "TOTAL": '#666666',
    "CT": '#b80058',
    "DISP": "#d95f02",
    "ELEC": '#5954d6',
    "POL": '#1b9e77',
    "PAULI": '#ebac23',
    'FRZ': '#a6cee3', 
    'ORB': '#27C488'
}

def plot(type, args, dfs, ax, cmap_range=[-100, 0.0], median=False, ratio1=False, ratio2=False, plot=False, kde_norm=False):
    """
    General Plot Function

    Parameters
    ----------
    type: {'ternary', 'box', 'dist', 'violin'}
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
        - 'box': (labels, items)
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
        - 'violin': (group_name, groups, plot)
            - group_name: str
                name of the column to group by
            - groups: list of str
                list of group names
            - plot: str, optional (default='violin')
                type of plot ('violin' or 'bar')
        - 'pie': (group_name, groups)
    df: pd.DataFrame or list of pd.DataFrame
        Interaction Dataframe with EDA energies
    ax: ax
        ax
    cmap_range: [int, int]
        color map energy range for ternary plot from lowest to highest
    median: bool
        plot median for box plot
    ratio1: bool
        ratio1 for box plot
    ratio2: bool
        ratio2 for box plot
    plot: bool
        plot lines for box plot
    kde_norm: bool
        normalize highest value of each distributions to one in dist plot
    Returns
    -------
    sc: ax
        plot
    """
    if type == 'box': #temp replace DISP with CLS_DISP for no_Orth_Decomposition
        for df in dfs:
            df.loc[df['DISP'].isna(), 'DISP'] = df['CLS_DISP']
    # else:
        # dfs.loc[dfs['DISP'].isna(), 'DISP'] = dfs['CLS_DISP']
    
    if type == 'ternary':
        label, index, type = args
        return ternary_ax(dfs, ax, title=f'({index}) {dfs.shape[0]} {label}', type=type, cmap_range=cmap_range)
    elif type == 'box':
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
    elif type == 'violin':
        group_name, groups, labels, plot = args
        violin(axes, group_name, groups, labels, df, plot)
    elif type == 'pie':
        group_name, groups, labels = args
        pie(axes, group_name, groups, labels, df)
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
    items = ['FROZEN', 'CLS_ELEC', 'MOD_PAULI', 'ELEC+PAULI', 'DISP', 'PAULI+DISP', 'POLARIZATION', 'CHARGE_TRANSFER', 'TOTAL']
    portion = {}

    def aggregate(data):
        if median:
            return data.median()
        else:
            return data.mean()
    
    for item in items:
        if item == 'PAULI+DISP':
            data = df['MOD_PAULI'] + df['DISP']
        elif item == 'ELEC+PAULI':
            data = df['CLS_ELEC'] + df['MOD_PAULI']
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


def violin(axes, group_name, groups, labels, processed_df, plot='violin', isFloat=False):
    components = [
        ('TOTAL', 'TOTAL'),
        ('MOD_PAULI', 'PAULI'),
        ('CLS_ELEC', 'ELEC'),
        ('DISP', 'DISP'),
        ('POLARIZATION', 'POL'),
        ('CHARGE_TRANSFER', 'CT')
    ]
    
    for i, (c, label) in enumerate(components):
        ax = axes.flatten()[i]
        processed_df_all = processed_df.copy()

        if 'ALL' in groups:
            processed_df_all[group_name] = 'ALL' 
            processed_df_all = pd.concat([processed_df, processed_df_all])

        if plot == 'violin':

            if len(groups) == 2:
                xticks = [-0.2, 0.2]
            elif len(groups) == 3:
                xticks = [-0.3, 0.0, 0.3]
            elif len(groups) == 4:
                xticks = [-0.34, -0.11, 0.11, 0.34]
            elif len(groups) == 5:
                xticks = [-0.36, -0.18, 0.0, 0.18, 0.36]

            sns.violinplot(processed_df_all, hue=group_name, y=c, ax=ax, density_norm='width', cut=0, split=False, inner=None, legend=False, gap=0.1, width=0.9, hue_order=groups)
            ax.set_xticks(xticks)
            ax.set_xticklabels(labels)
        elif plot == 'bar':

            if len(groups) == 2:
                xticks = [0, 1]
            elif len(groups) == 3:
                xticks = [0, 1, 2]
            elif len(groups) == 4:
                xticks = [0, 1, 2, 3]
            elif len(groups) == 5:
                xticks = [0, 1, 2, 3, 4]

            for n, group in enumerate(groups):
                if isFloat:
                    subdf = processed_df_all.query(f"{group_name} == {group}")
                else:
                    subdf = processed_df_all.query(f"{group_name} == '{group}'")
                data = subdf[c]
                err = np.abs(np.percentile(data, 75) - np.percentile(data, 25)) / 2
                ax.bar(n, np.median(data), yerr=err, color=COLOR_MAPS[label], alpha=0.9, width=0.5, capsize=3, error_kw={'elinewidth': 1.5})
            ax.set_xticks(xticks)
            ax.set_xticklabels(labels)
        
        ax.set_title(label)
        for index in range(len(ax.collections)):
            ax.collections[index].set_facecolor(COLOR_MAPS[label])
            ax.collections[index].set_alpha(1.0 - (0.2 * index))
        ax.set_axisbelow(True)
        ax.grid(True, axis='y', which='both')
        ax.set_ylabel('')
        ax.tick_params(direction='in', which='both', axis='y')

    return axes

def pie(axes, group_name, groups, labels, processed_df, type='TOTAL', isFloat=False):
    def compute_fraction(df):
        elec = np.mean(df['CLS_ELEC'])
        pauli = np.mean(df['MOD_PAULI'])
        disp = np.mean(df['DISP'])
        pol = np.mean(df['POLARIZATION'])
        ct = np.mean(df['CHARGE_TRANSFER'])
        total = np.mean(df['TOTAL'])

        orb = pol + ct
        stab = orb + elec + disp
        frz = elec + disp + pauli

        if type == 'TOTAL':
            return [frz/total, pol/total, ct/total]
        elif type == 'STABILIZING':
            return [elec/stab, disp/stab, pol/stab, ct/stab]

    def find_label(x, data):
        if type == 'TOTAL':
            pcts = [data[0]/sum(data)*100, data[1]/sum(data)*100, data[2]/sum(data)*100]
            if abs(pcts[0] - x) < 1e-5:
                return f'FRZ\n({x:.2f}%)'
            elif abs(pcts[1] - x) < 1e-5:
                return f'POL\n({x:.2f}%)'
            elif abs(pcts[2] - x) < 1e-5:
                return f'CT\n({x:.2f}%)'
            else:
                return f'{x:.2f}%'
        elif type == 'STABILIZING':
            pcts = [data[0]/sum(data)*100, data[1]/sum(data)*100, data[2]/sum(data)*100, data[3]/sum(data)*100]
            if abs(pcts[0] - x) < 1e-5:
                return f'ELEC\n({x:.2f}%)'
            elif abs(pcts[1] - x) < 1e-5:
                return f'DISP\n({x:.2f}%)'
            elif abs(pcts[2] - x) < 1e-5:
                return f'POL\n({x:.2f}%)'
            elif abs(pcts[3] - x) < 1e-5:
                return f'CT\n({x:.2f}%)'
            else:
                return f'{x:.2f}%'
        
    if type == 'TOTAL':
        kwargs = dict(
            colors=[COLOR_MAPS['FRZ'], COLOR_MAPS['POL'], COLOR_MAPS['CT']], 
            textprops=dict(fontsize=12),
            wedgeprops = {'linewidth': 3, 'clip_on': True},
            autopct=lambda p: find_label(p, data)
        )
    elif type == 'STABILIZING':
        kwargs = dict(
            colors=[COLOR_MAPS['ELEC'], COLOR_MAPS['DISP'], COLOR_MAPS['POL'], COLOR_MAPS['CT']], 
            textprops=dict(fontsize=12),
            wedgeprops = {'linewidth': 3, 'clip_on': True},
            autopct=lambda p: find_label(p, data)
        )

    processed_df_all = processed_df.copy()
    if 'ALL' in groups:
        processed_df_all[group_name] = 'ALL' 
        processed_df_all = pd.concat([processed_df, processed_df_all])

    if len(groups) > 1:
        for i in range(len(groups)):
            if isFloat:
                data = compute_fraction(processed_df_all.query(f"{group_name} == {groups[i]}"))
            else:
                data = compute_fraction(processed_df_all.query(f"{group_name} == '{groups[i]}'"))
            axes[i].pie(data, **kwargs)
            axes[i].set_title(labels[i], fontsize=14)
            axes[i].set_xlim(-1, 1)
            axes[i].set_ylim(-1.1, 1.1)
    elif len(groups) == 1:
        if isFloat:
            data = compute_fraction(processed_df_all.query(f"{group_name} == {groups[0]}"))
        else:
            data = compute_fraction(processed_df_all.query(f"{group_name} == '{groups[0]}'"))
        axes.pie(data, **kwargs)
        axes.set_title(labels[0], fontsize=14)
        axes.set_aspect('equal')

    return axes
    
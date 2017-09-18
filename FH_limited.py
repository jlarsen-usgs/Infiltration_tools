import sys
if sys.version_info[0] < 3:
    raise EnvironmentError(
        "Python 3 must be used for filtering data, "
        "set filter to False to use python 2.7")

import csv
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter

from scipy.signal import savgol_filter
from scipy.stats import linregress
from scipy.optimize import minimize


def get_cumulative_infiltration(time_depth):
    inf_data = []
    for idx, record in enumerate(time_depth):
        if idx == 0:
            I0 = 0
            I = 0
            i = 0
        else:
            I = I0 + time_depth[idx - 1][-1] - record[-1]
            i = I/idx

        I0 = I
        inf_data.append([idx, record[-1], I, i])

    return np.array(inf_data)


def filter_data(time_depth):
    depth = np.array([i[-1] for i in time_depth])
    filtered_depth = savgol_filter(depth, 9, 1)

    return np.array([[idx, i] for idx, i in enumerate(filtered_depth)])


def create_tables(inf_test, irfile):
    """
    User method to create matplotlib tables of processed data
    """

    header = (r'$Test$', r'$i\ cm/s$', r'$i\ r^{2}$',
              r'$S\  cm/s^{1/2}$', r'$\beta$', r'$n_{eff}$', r'$K_s\ cm/s$')

    dm = []


    dm.append(("Falling Head",
               "{:.2e}".format(inf_test['ss_i']),
               "{:.2e}".format(inf_test['r2']),
               "---",
               "---",
               "---",
               "{:.2e}".format(inf_test['ks'])))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')

    table = ax.table(cellText=dm,
                     colLabels=header,
                     loc='center',
                     fontsize=12)

    table_props = table.properties()
    table_cells = table_props['child_artists']
    for cell in table_cells: cell.set_height(0.075)

    fig.tight_layout()

    plt.savefig("{}_table1.png".format(irfile))
    plt.close()

    header1 = (r'$\bar\beta$', r'$\bar n_{eff}$', r'$\sigma\ n_{eff}$',
               r'$\bar i\ cm/s$', r'$\sigma\ i$', r'$\bar K_s\ cm/s$',
               r'$\sigma\ K_s$', r'$K_{ss}\ cm/s$', r'$L_f\ cm$')

    dm1 = [('---',
            '---',
            '---',
            '{:.2e}'.format(inf_test['ss_i']),
            '---',
            '{:.2e}'.format(inf_test['ks']),
            '---',
            '{:.2e}'.format(inf_test['ks']),
            '---')]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')

    table = ax.table(cellText=dm1,
                     colLabels=header1,
                     loc='center',
                     fontsize=12)

    table_props = table.properties()
    table_cells = table_props['child_artists']
    for cell in table_cells: cell.set_height(0.075)

    fig.tight_layout()

    plt.savefig("{}_table2.png".format(irfile))

    with open("{}_analysis.txt".format(irfile), 'w') as af:
        af.write('Test,i cm,i r^2,S cm/s^1/2,beta,n_eff,K_s cm/s\n')
        for test in dm:
            s = ",".join([str(i) for i in test])
            s += "\n"
            af.write(s)
        af.write("\n")

        af.write('mean beta,mean n_eff,std n_eff,mean i cm/s,std i,'
                 'mean K_s cm/s,std K_s,K_ss cm/s,L_f cm\n')

        s = ",".join([str(i) for i in dm1[0]])
        af.write(s)

if __name__ == '__main__':
    with open('Inf5_fh.csv') as inf:
        td = []
        reader = csv.reader(inf)
        for idx, line in enumerate(reader):
            if idx == 0:
                pass
            else:
                td.append([idx-1, float(line[4]) * 30.48])

td = filter_data(td)
inf_data = get_cumulative_infiltration(td)

inf_data = inf_data.T
regress = linregress(inf_data[0], inf_data[2])
ks = regress.slope * (3/2)

ss_i = regress.slope
ss_b = regress.intercept
rsq = regress.rvalue ** 2

reg_data = np.array([[0, ss_b], [inf_data[0, -1], ss_i * inf_data[0, -1] + ss_b]])
reg_data = reg_data.T

plt.plot(inf_data[0], inf_data[2], 'g')
plt.plot(reg_data[0], reg_data[1], 'g--')
plt.ylabel(r"$cm$", fontsize=18)
plt.xlabel(r"$s$", fontsize=18)
plt.savefig("ks_ssi_Inf5.csv.png")
plt.close()

plt.plot(inf_data[0], inf_data[-1], 'g')
plt.ylabel(r"$cm/s$", fontsize=18)
plt.xlabel(r"$cm/s$", fontsize=18)
plt.savefig("i_Inf5.csv.png")
plt.close()

inf_data = {'ss_i': ss_i, 'r2': rsq, 'ks': ks}
create_tables(inf_data, "Inf5.csv")


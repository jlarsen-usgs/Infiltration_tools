import matplotlib.pyplot as plt
import os
from collections import OrderedDict


def read_analysis_file(afile):
    read_data = False
    data = {}
    with open(afile) as af:
        for line in af:
            if line.startswith("mean beta"):
                read_data = True
            elif read_data:
                t = line.strip('\n').split(',')
                data = {"beta": t[0], 'n': (t[1], t[2]), 'i': (t[3], t[4]),
                        'ks': (t[5], t[6], t[7]), 'lf': t[8]}
            else:
                pass

    return data


def create_tables(inf_data):
    """
    User method to create matplotlib tables of processed data
    """

    header1 = (r'$Test$', r'$\bar\beta$', r'$\bar n_{eff}$',
               r'$\sigma\ n_{eff}$',
               r'$\bar i\ cm/s$', r'$\sigma\ i$', r'$\bar K_s\ cm/s$',
               r'$\sigma\ K_s$', r'$K_{ss}\ cm/s$', r'$L_f\ cm$')
    dm1 = []
    ll = []
    for key, value in inf_data.items():
        dm1.append(('{}'.format(key),
                    value['beta'],
                    value['n'][0],
                    value['n'][1],
                    value['i'][0],
                    value['i'][1],
                    value['ks'][0],
                    value['ks'][1],
                    value['ks'][2],
                    value['lf']))
        ll.append((value['lat'], value['lon']))

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
    plt.savefig("Table_all_tests.png")

    with open("Analysis.txt", 'w') as af:

        af.write('Test,mean beta,mean n_eff,std n_eff,mean i cm/s,std i,'
                 'mean K_s cm/s,std K_s,K_ss cm/s,L_f cm,Latitude,Longitude\n')
        for idx, val in enumerate(dm1):
            s = ",".join([str(i) for i in val])
            s += "{},{}".format(*ll[idx])
            af.write(s)
            af.write('\n')


if __name__ == "__main__":
    lat_lon = [(34.762167, -120.156639),
               (34.743389, -120.271278),
               (34.745611, -120.287667),
               (34.730427, -120.279195),
               (34.780967, -120.345788),
               (34.755781, -120.349069),
               (34.759056, -120.394222),
               (34.771472, -120.428056),
               (34.7585, -120.43175),
               (34.811861, -120.449),
               (34.745953, -120.280863),
               (34.767153, -120.433630)]
    ws = 'Full'
    test_base = 'Inf{}.csv_analysis.txt'
    name_base = "Inf {}"
    tests = [test_base.format(i) for i in range(12) if i not in (0, 2)]
    names = [name_base.format(i) for i in range(12) if i not in (0, 2)]

    tests += ['Trod2.csv_analysis.txt', 'Trod3.csv_analysis.txt']
    names += ["Trod 2", "Trod 3"]

    data = OrderedDict()
    for idx, fname in enumerate(tests):
        data[names[idx]] = read_analysis_file(os.path.join(ws, fname))

    i = 0
    for key, value in data.items():
        data[key]['lat'] = lat_lon[i][0]
        data[key]['lon'] = lat_lon[i][1]

    create_tables(data)
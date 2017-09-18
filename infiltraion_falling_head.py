"""
__author__: Joshua D Larsen

Infiltration toolbox piece to analyze falling head infiltration data
from multiple falling head tests.

Can return estimates of sorptivity, hydraulic conductivity, and SWC curves
from infiltration tests.

This program uses lots of math!
"""
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


class FallingHeadOnlyTests(object):
    """
    Class that seperates curves and builds out falling head infiltration test
    curves

    Parameters:
        pt_file: (str) win_situ pressure transducer file name
        pt_unit: (str) base units of the pressure transducer
        sensitivity: (float) sensitivity to change in data for splitting curves
            (cm)
        ignore_test: (tuple) test number(s) to ignore if data is bad
        filter: (bool) flag to indicate whether to apply savgol golay filtering
            to raw data (recommended!)
    """
    def __init__(self, pt_file, pt_unit='ft', sensitivity=3.048,
                 ignore_test=(), filter=True):
        self.__file = pt_file
        self.data_startline = 0
        self.pt_unit = pt_unit
        self.time_index = 0
        self.depth_index = 0
        self.header = None
        self.raw_data = []
        self.raw_infiltration = {}
        self.filtered_infiltration = {}
        self.sensitivity = sensitivity
        self.sorptivity = {}
        self.ks = {}
        self.unsaturated = {}
        self.colors = ["k", "maroon", "b", "darkolivegreen",
                       "darkgrey", "c", "m"]

        t = self.unit_to_cm

        self.__filter = filter
        self.__time_depth = []
        self.__ignore_test = ignore_test

        self.__get_raw_data()
        self.__get_time_depth_data()
        self.__get_raw_infiltration_curves()

        if filter:

            self.__filter_data()

    def calculate_ks(self, precision=0.005, bnds=((0., None), (0.01, 0.35))):
        """
        Use Woodings equation to calcuate 3d infiltration and recover saturated
        hydraulic conductivity for each test. We first find the asymptote that
        defines steady state infiltration by monitoring the change in cumulative
        infiltration/time step

        Parameters:
            precision: (float) percentage change in the short term infiltration
                rate that defines asymptotic behaviour.
            bnds: (float) bounds for ks, and effective porosity when using
                least sqares method of solution
        """
        if not self.sorptivity:
            self.calculate_sorptivity()

        if self.__filter:
            inf_data = self.filtered_infiltration
        else:
            inf_data = self.raw_infiltration

        for key, inf_test in inf_data.items():
            # get steady state infiltration information for the caculation of ks
            ks_data = np.recarray((1,), dtype=inf_test.dtype)
            asymp_index, asymp_precision = self.__find_asymptote(inf_test,
                precision=precision)

            for idx, record in enumerate(inf_test[asymp_index:]):
                if idx == 0:
                    ks_data[0] = tuple(record)
                else:
                    ks_data.resize((len(ks_data) + 1,),
                                           refcheck=False)
                    ks_data[-1] = tuple(record)

            regress = linregress(ks_data['dt'], ks_data['I'])
            ss_i = regress[0]
            ss_b = regress[1]
            r_sq = regress[2] **2

            ss_i_curve = np.recarray((2,), dtype=[('dt', np.float),
                                                  ('I', np.float)])

            ss_i_curve[0] = (inf_test['dt'][0],
                             ss_i * inf_test['dt'][0] + ss_b)

            ss_i_curve[-1] = (inf_test['dt'][-1],
                              ss_i * inf_test['dt'][-1] + ss_b)

            ks, n = self.__calculate_ks(self.sorptivity[key]['S'],
                                        ss_i, bnds=bnds)

            self.ks[key] = {'ks': ks, 'ss_data': ks_data,
                            'ss_curve': ss_i_curve, 'ss_i': ss_i,
                            'n': n, 'r2': r_sq, 'b': 0.55}
        return self.ks

    def calculate_unsaturated(self, mean_ks=True):
        """
        User method to calculate unsaturated zone parameters from infiltration
        data, including depth of wetting front, soil water characteristic,
        and brooks corey parameters.

        Parameters:
            mean_ks: (bool) if True calculate curves with mean ks,
                if False use late time ks.

        Returns:
            self.unsaturated (dict) unsaturated zone parameters.
        """
        if not self.sorptivity:
            self.calculate_sorptivity()

        if not self.ks:
            self.calculate_ks()

        if self.__filter:
            inf_data = self.filtered_infiltration
        else:
            inf_data = self.raw_infiltration

        ks = []
        ss_i = []
        s = []
        b = []
        n = []
        h = []
        t = []
        test_no = 1
        for key, inf_test in sorted(inf_data.items()):
            ks.append(self.ks[key]['ks'])
            ss_i.append(self.ks[key]['ss_i'])
            s.append(self.sorptivity[key]['S'])
            b.append(self.ks[key]['b'])
            n.append(self.ks[key]['n'])
            h.append(inf_test['depth'][0])
            if test_no == 1:
                t.append(inf_test['time'][0])
            else:
                t.append(inf_test['time'][-1])
            test_no += 1

        if not mean_ks:
            ga_ks = ks[-1]
        else:
            ga_ks =  np.mean(ks)

        ks_std = np.std(ks)
        ks = np.mean(ks)
        i_std = np.std(ss_i)
        ss_i = np.mean(ss_i)
        ga_factor = 1. - (ks/ss_i)
        s = np.mean(s)
        b = np.mean(b)
        n_std = np.std(n)
        n = np.mean(n)
        h = np.sum(h)
        t = dt.datetime.combine(dt.date.min, t[-1]) -\
            dt.datetime.combine(dt.date.min, t[0])
        t = t.seconds



        beta = self.__estimate_gardner_beta(n, ks, b, s)
        gardner_kh = self.__estimate_kh(ks, beta)

        wetting_front = self.__estimate_wetting_front_depth(10., h, ga_ks,
                                                            t, n, ga_factor)

        continuous = self.__create_continuous_infiltration_array()

        wf_advance = np.recarray((1,), dtype=[('dt', np.float),
                                              ('delh', np.float),
                                              ('Lf', np.float)])

        for idx, record in enumerate(continuous):
            if idx == 0:
                lf = self.__estimate_wetting_front_depth(0.1, record['delh'],
                                                         ga_ks, record['dt'],
                                                         n, ga_factor)

                wf_advance[0] = (record['dt'], record['delh'], lf)

            else:
                lf = self.__estimate_wetting_front_depth(wf_advance['Lf'][-1],
                                                         record['delh'], ga_ks,
                                                         record['dt'], n,
                                                         ga_factor)

                wf_advance.resize((len(wf_advance) + 1,), refcheck=False)
                wf_advance[-1] = (record['dt'], record['delh'], lf)


        lf2 = (continuous['I']/n) * (1. - ga_factor)
        wf_advance2 = np.recarray((len(lf2),), dtype=[('dt', np.float),
                                                      ('Lf', np.float)])
        for idx, rec in enumerate(lf2):
            wf_advance2[idx] = (idx, rec)

        self.unsaturated = {'beta': beta, 'gardner_curve': gardner_kh,
                            'wetting_front': wf_advance2[-1]['Lf'],
                            'n': (n, n_std), 'ks': (ks, ks_std),
                            'ss_i': (ss_i, i_std), 'lf_advance': wf_advance,
                            'lf_advance2': wf_advance2}

        return self.unsaturated

    def calculate_sorptivity(self, sqrt_dt=3):
        """
        Use the early time Phillips equation solution to get an estimate
        of Sorptivity for infiltration tests.

        Parameters:
            sqrt_dt: (float) time cutoff for the calcuation of sorptivity

        Returns:

        """

        if self.__filter:
            inf_data = self.filtered_infiltration
        else:
            inf_data = self.raw_infiltration

        for key, inf_test in inf_data.items():
            sorptivity_data = np.recarray((1,), dtype=inf_test.dtype)
            sorptivity_data[0] = tuple(inf_test[0])
            for idx, record in enumerate(inf_test):
                if record['sqrt_dt'] < sqrt_dt and idx != 0:
                    sorptivity_data.resize((len(sorptivity_data) + 1,),
                                           refcheck=False)
                    sorptivity_data[-1] = tuple(record)

                elif record['sqrt_dt'] == sqrt_dt:
                    sorptivity_data.resize((len(sorptivity_data) + 1,),
                                           refcheck=False)
                    sorptivity_data[-1] = tuple(record)
                    break

                elif idx != 0:
                    sorptivity_data.resize((len(sorptivity_data) + 1,),
                                           refcheck=False)
                    sorptivity_data[-1] = tuple(inf_test[idx - 1])
                    break

                else:
                    pass

            t1 = sorptivity_data["I"]
            t2 = sorptivity_data['sqrt_dt']
            regress = linregress(sorptivity_data['sqrt_dt'],
                                 sorptivity_data['I'])

            S = regress[0]  # sorptivity is the slope of the regression curve
            b = regress[1]  # intercept for modeling sorptivity regression

            sorptivity_curve = np.recarray((2,), dtype=[('sqrt_dt', np.float),
                                                        ('I', np.float)])
            sorptivity_curve[0] = (sorptivity_data[0]['sqrt_dt'],
                                   S * inf_test[0]['sqrt_dt'] +
                                   b)

            sorptivity_curve[-1] = (inf_test[-1]['sqrt_dt'],
                                    S * inf_test[-1]['sqrt_dt'] +
                                    b)

            self.sorptivity[key] = {"S": S, "S_data": sorptivity_data,
                                    "S_curve": sorptivity_curve}

        return self.sorptivity

    def __estimate_gardner_beta(self, n, ks, b, s):
        """
        Method to calculate Gardner 1958 beta function for each test

        Parameters:
            n (float): theta_f - theta_i (effective porosity)
            ks (float) saturated hydraulic conductivity
            b (float) shape factor defaults to 0.55
            s (float) sorptivity
        """
        return (n * ks)/(b * s**2)

    def __estimate_kh(self, ks, beta):
        """
        Use Gardener 1958 to estimate kh with regard to matric potential

        Parmaeters:
            ks: (float) saturated hydraulic conductivity
            beta: (float) Gardener's beta

        Returns:
            kh_curve
        """

        h = np.logspace(np.log10(1), np.log10(10**6), num=1000)
        kh = ks * np.exp(beta * -h)

        # filter out trailing zero values where -h is now too great
        for idx, value in enumerate(kh):
            if value == 0:
                kh = kh[:idx]
                h = h[:idx]
                break

        kh_curve = np.recarray((len(h),), dtype=[('h', np.float),
                                                 ('kh', np.float)])

        for idx, val in enumerate(h):
            kh_curve[idx] = (val, kh[idx])

        return kh_curve

    def __estimate_wetting_front_depth(self, lf, delh, ks, t, n, ga_factor):
        """
        Green-ampt equation wrapped in the Newton-Rahpson solver to
        solve for depth of the wetting front during infiltration test.

        Parameters:
            lf: (float) inital guess at wetting front depth
            delh: (float) total depth of water applied to surface during tests
            ks: (float) hydraulic conductivity
            t: (float) total time of tests
            n: (float) effective porosity (theta_f - theta_i)
            ga_factor: (float) scaling factor to account for 3d radial flow.
                Calculated as 1 - ks / infiltration_rate. Literally fraction
                of infiltration lost to radial flow assuming late time
                infiltration.
        """
        f = lf - delh * np.log(1 + lf/delh) - (ks * t)/n
        f_prime = lf / (delh + lf)

        lf1 = (1. - ga_factor) * (lf - (f/f_prime))

        if abs(lf1 - lf) > 0.01:
            lf1 = self.__estimate_wetting_front_depth(lf1, delh, ks,
                                                      t, n, ga_factor)

        return lf1

    def __create_continuous_infiltration_array(self):
        """
        Creates a continuous function infiltration array to estimate depth
        of wetting front vs. time.
        """
        if self.filtered_infiltration:
            inf_data = self.filtered_infiltration
        else:
            inf_data = self.raw_infiltration

        continuous = np.recarray((1,), dtype=[('dt', np.float),
                                              ('delh', np.float),
                                              ('I', np.float)])
        test_no = 1
        t0 = None
        for key, inf_test in sorted(inf_data.items()):
            break_idx = len(inf_test) - 1
            for idx, record in enumerate(inf_test):
                if test_no == 1:
                    if idx == 0:

                        t0 = record['time']
                        d0 = record['depth']

                    else:
                        delt = dt.datetime.combine(dt.date.min, record['time']) - \
                               dt.datetime.combine(dt.date.min, t0)
                        delt = delt.seconds
                        deld = d0 - record['depth']

                        if idx==1:
                            continuous[0] = (delt, deld, record['I'])

                        else:
                            continuous.resize((len(continuous) + 1,),
                                              refcheck=False)
                            continuous[-1] = (delt, deld, record['I'])

                else:
                    if idx == 0:
                        tbe = dt.datetime.combine(dt.date.min, record['time'])
                        delta = dt.timedelta(seconds=1)
                        tb = dt.datetime.combine(dt.date.min, tb) +  delta
                        d0 = record['depth']
                        dcum = continuous[-1]['delh']
                        while tb <= tbe:
                            delt = tb - dt.datetime.combine(dt.date.min, t0)
                            delt = delt.seconds
                            deld = continuous[-1]['delh']
                            continuous.resize((len(continuous) + 1,),
                                              refcheck=False)
                            continuous[-1] = (delt, deld, I0)
                            tb += delta

                    else:
                        delt = dt.datetime.combine(dt.date.min,
                                                   record['time']) - \
                               dt.datetime.combine(dt.date.min, t0)
                        delt = delt.seconds
                        deld = dcum + (d0 - record['depth'])
                        continuous.resize((len(continuous) + 1,),
                                          refcheck=False)
                        continuous[-1] = (delt, deld, I0 + record['I'])

                if idx == break_idx:
                    tb = record['time']
                    I0 = continuous["I"][-1]
                    test_no += 1

        return continuous

    def __calculate_ks(self, sorptivity, ss_i, b=0.55, rs=30.48,
                       bnds=((0., None), (0.01, 0.35))):
        """
        Function to begin caculation of ks through minimization
        routine. We solve for unknown ks and n (theta_f - theta_i).
        Using the Wooding 1968 for ks calculation.

        Parameters:
            sorptivity: (float) sorptivity value from regression
            ss_i: (float) steady state infiltration rate
            b: (float) shape factor from wooding 1968
            rs: (float) radius of the inner infiltrometer ring in cm
            bnds: (tuple) 2x: first tuple is a Ks bound, second is for
                theta_f - theta_i (n)

        Returns:

        """
        x = minimize(self.__ks_residual, [0.001, 0.2],
                     args=(sorptivity, ss_i, b, rs),
                     method='SLSQP', bounds=bnds)
        return x['x']

    def __ks_residual(self, x, sorptivity, ss_i, b,
                      rs):
        """
        Set up the residual function for determining ks and n. This function
        allows for minimization through Sequential Least sqares.

        Parameters:
            x (list) initial guess at [ks, n]
            sorptivity: (float) sorptivity value from regression
            ss_i: (float) steady state infiltration rate
            b: (float) shape factor from wooding 1968
            rs: (float) radius of the inner infiltrometer ring in cm

        Returns:
            residual (float)
        """
        ks, n = x
        return abs(ss_i - ((4. * b * sorptivity ** 2) / (np.pi * rs * n)) - ks)

    def __find_asymptote(self, inf_test, precision=0.005):
        """
        Finds the asymptote of steady state infiltration for Ks
        Parameters:
            inf_test: (recarray) infiltration test data
            precision: (float) percent/fraction of variance to define asymptote

        Returns:
            asymp_idx: (int) index of the beginning of asymptotic behaviour
        """
        i = []
        asymp = []
        for idx, record in enumerate(inf_test):
            if idx == 0:
                pass
            elif idx == 1:
                i0 = record['I'] / record['dt']
                i.append(i0)
            else:
                dt = (record['dt'] - inf_test[idx - 1]['dt'])
                i0 = record['I'] / record['dt']
                delta_percent = abs(i0 - i[-1]) / i[-1]

                i.append(i0)
                asymp.append(delta_percent)

        asymp_idx = None
        asymp_finder = [False] * 20
        for idx, value in enumerate(asymp):
            if value < precision:
                asymp_finder.append(True)
            else:
                asymp_finder.append(False)

            asymp_finder.pop(0)

            asymp_found = True
            for i in asymp_finder:
                if not i:
                    asymp_found = False

            if asymp_found:
                asymp_idx = idx - 18
                break

        if asymp_idx is None:
            asymp_idx, precision = self.__find_asymptote(inf_test,
                precision=precision+0.001)
        return asymp_idx, precision

    def __filter_data(self):
        """
        Method applies a 9 point Savgol Golay filter to infiltration data
        and sets data to self.filtered_infiltration
        """
        for key, inf_test in self.raw_infiltration.items():
            filtered_inf = np.recarray((len(inf_test),),
                                       dtype=np.dtype([('time', np.object),
                                                       ('dt', np.float),
                                                       ('sqrt_dt', np.float),
                                                       ('depth', np.float),
                                                       ('I', np.float),
                                                       ('i', np.float)]))

            depth = np.array([i for i in inf_test['depth']])
            filtered_depth = savgol_filter(depth, 9, 1)

            d0 = depth[0]
            for idx, rec in enumerate(inf_test):
                I = d0 - filtered_depth[idx]
                try:
                    i = I/rec['dt']
                except (ZeroDivisionError, RuntimeWarning):
                    i = 0.

                filtered_inf[idx] = (rec['time'], rec['dt'], rec['sqrt_dt'],
                                     filtered_depth[idx], I, i)

            self.filtered_infiltration[key] = filtered_inf

    def __get_raw_infiltration_curves(self):
        """
        Method to determine the beginning and end of each infiltration test
        and log the data to self.raw_infiltration by test
        """
        curves = {}
        test_begins = False
        logging_test = False
        test_no = 1
        test = []
        for idx, line in enumerate(self.__time_depth):
            if idx == 0:
                pass
            else:
                delta_depth = line[-1] - self.__time_depth[idx - 1][-1]

                if delta_depth >= self.sensitivity and\
                        line[-1] >= self.sensitivity:
                    test_begins = True

                elif test_begins:
                    if delta_depth < 0:
                        test = [self.__time_depth[idx - 1], line]
                        logging_test = True
                        test_begins = False

                elif logging_test:
                    test.append(line)
                    if line[-1] <= 0.:
                        logging_test = False
                        adj_idx = 0
                        max_val = test[0][-1]
                        for index, line in enumerate(test):
                            if line[-1] > max_val:
                                adj_idx = index
                                max_val = line[-1]
                        curves[test_no] = test[adj_idx:]
                        test = []
                        test_no += 1

        if logging_test:
            adj_idx = 0
            max_val = test[0][-1]
            for index, line in enumerate(test):
                if line[-1] > max_val:
                    adj_idx = index
                    max_val = line[-1]
            curves[test_no] = test[adj_idx:]

        if self.__ignore_test:
            for test_no in self.__ignore_test:
                try:
                    curves.pop(test_no)
                except KeyError:
                    raise AssertionError("Test no: {} was not found in "
                                         "data, please check test counts or "
                                         "adjust sensitivity".format(test_no))

        for key, inf_test in curves.items():
            infiltration = np.recarray((len(inf_test),),
                                       dtype=np.dtype([('time', np.object),
                                                       ('dt', np.float),
                                                       ('sqrt_dt', np.float),
                                                       ('depth', np.float),
                                                       ('I', np.float),
                                                       ('i', np.float)]))
            t0, d0 = inf_test[0]
            for idx, inf_rec in enumerate(inf_test):
                delt = dt.datetime.combine(dt.date.min, inf_rec[0]) -\
                       dt.datetime.combine(dt.date.min, t0)
                delt = delt.seconds
                sqrt_time = np.sqrt(delt)
                I = d0 - inf_rec[-1]
                try:
                    i = I/float(delt)
                except ZeroDivisionError:
                    i = 0.
                infiltration[idx] = (inf_rec[0], delt, sqrt_time,
                                     inf_rec[-1], I, i)

            self.raw_infiltration[key] = infiltration

    def __get_time_depth_data(self):
        """
        Background method that grabs the time object and depth of ponding
        """
        time_depth = []
        for record in self.raw_data:
            try:
                time = self.__format_time(record[self.time_index])
                depth = float(record[self.depth_index]) * self.unit_to_cm
                time_depth.append([time, depth])
            except ValueError:
                pass
        self.__time_depth = time_depth

    def __format_time(self, s):
        """
        Format time method to create a datetime.time object from supplied str
        """
        t = s.strip().split(' ')

        pm = False
        if t[-1].lower() == "pm":
            pm = True

        t = t[1].split(':')
        if pm:
            if int(t[0]) != 12:
                return dt.time(hour=int(t[0]) + 12,
                               minute=int(t[1]),
                               second=int(t[2]))

        return dt.time(hour=int(t[0]), minute=int(t[1]), second=int(t[2]))

    def __get_raw_data(self):
        """
        Method to analyze the csv file and grab only the pressure transducer
        data from it
        """
        self.__get_data_startline()
        raw_data = []
        with open(self.__file) as ptf:
            reader = csv.reader(ptf)
            for idx, line in enumerate(reader):
                if idx == self.data_startline - 1:
                    self.header = [i for i in line if i != '']
                    self.__analyze_header()

                elif idx >= self.data_startline:
                    try:
                        t = [line[i] for i in range(len(self.header))]
                        raw_data.append(t)
                    except IndexError:
                        pass

                else:
                    pass

        self.raw_data = raw_data

    def __get_data_startline(self):
        """
        Method to find which line pt data begins in win situ csv files
        """
        data0 = False
        data1 = False
        with open(self.__file) as ptf:
            reader = csv.reader(ptf)
            for idx, line in enumerate(reader):
                try:
                    float(line[1])
                    float(line[2])
                    float(line[3])
                    float(line[4])
                    if data0:
                        data1 = True
                    else:
                        data0 = True
                except (IndexError, ValueError):
                    data0 = False
                    data1 = False

                if data0 and data1:
                    self.data_startline = idx - 1
                    break

        if not data0 and not data1:
            raise Exception('Data start line could not be found')

    def __analyze_header(self):
        """
        Method to analyze win-situ csv file headers
        """
        time = None
        depth = None
        for idx, head in enumerate(self.header):
            if 'time' in head.lower():
                time = idx
            elif 'depth' in head.lower():
                depth = idx
            else:
                pass

        if time is None or depth is None:
            raise AssertionError('Cannot find time or depth in '
                                 'transducer header')

        else:
            self.time_index = time
            self.depth_index = depth

    @property
    def unit_to_cm(self):
        if self.pt_unit.strip().rstrip().lower() not in ('ft', 'in',
                                                         'm', 'cm', 'mm', 'um'):
            raise AssertionError('pt_unit must be one of the following: ( ft ,'
                                 ' in , m , cm, mm , um )')
        conversions = {'ft': 30.48,
                       'in': 2.54,
                       'm': 100,
                       'cm': 1,
                       'mm': 0.1,
                       'um': 1e-4}
        return conversions[self.pt_unit.strip().rstrip().lower()]

    def plot_infiltration(self, data='depth', time='dt', test=(),
                          filtered=True, ax=None, savefig=False,
                          *args, **kwargs):
        """
        User method to plot infiltration data including water depth,
        infiltration rate, and cumulative infiltration vs time or sqrt time

        Args:
            data:
            time:
            test:
            filtered:
            ax:
            savefig:
            *args:
            **kwargs:

        Returns:
            matplotlib axis object
        """
        if filtered and self.filtered_infiltration:
            inf_data = self.filtered_infiltration
        else:
            inf_data = self.raw_infiltration

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if not test:
            for key, inf_test in inf_data.items():
                ax.plot(inf_test[time], inf_test[data],
                        label=r'${}\ {}$'.format(data, key),
                        color=self.colors[key],
                        *args, **kwargs)

        else:
            for key in test:
                try:
                    inf_test = inf_data[key]
                    ax.plot(inf_test[time], inf_test[data],
                            label=r'${}\ {}$'.format(data, key), *args, **kwargs)
                except KeyError:
                    pass

        ax.legend(loc=0, fontsize=12)

        if data == 'i':
            ax.set_ylabel(r'$cm/s$', fontsize=20)
        else:
            ax.set_ylabel(r'$cm$', fontsize=20)

        if time == "sqrt_dt":
            ax.set_xlabel(r'$s^{1/2}$', fontsize=20)
        else:
            ax.set_xlabel(r'$s$', fontsize=20)

        if savefig:
            plt.savefig("{}_{}.png".format(data, self.__file))
            plt.close()

        else:
            return ax

    def plot_sorptivity(self, data="I", time="sqrt_dt", test=(),
                        ax=None, savefig=False, *args, **kwargs):
        """
        User method to plot sorptivity curve used to determine sorptivity

        Args:
            data:
            time:
            test:
            ax:
            savefig:
            *args:
            **kwargs:

        Returns:
            matplotlib axis object
        """
        if data != 'I' and time != "sqrt_dt":
            raise AssertionError("Data must equal I and "
                                 "time must equal sqrt_dt")

        inf_data = self.sorptivity

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if not test:
            for key, inf_test in inf_data.items():
                ax.plot(inf_test['S_curve'][time], inf_test['S_curve'][data],
                        label=r'$S\ {}$'.format(key), color=self.colors[key],
                        *args, **kwargs)

        else:
            for key in test:
                try:
                    inf_test = inf_data[key]
                    ax.plot(inf_test['S_curve'][time],
                            inf_test['S_curve'][data],
                            label=r'$S\ {}$'.format(key), *args, **kwargs)
                except KeyError:
                    pass

        ax.set_ylabel(r'$cm$', fontsize=20)
        ax.set_xlabel(r'$s^{1/2}$', fontsize=20)

        if savefig:
            plt.legend(loc=0, fontsize=12)
            plt.savefig("S_{}.png".format(self.__file))
            plt.close()

        else:
            return ax

    def plot_ks(self, data='I', time='dt', test=(),
                ax=None, savefig=False, *args, **kwargs):
        """
        User method to plot steady state infiltration data used to calculate
        ks.
        Args:
            data:
            time:
            test:
            ax:
            savefig:
            *args:
            **kwargs:

        Returns:
            matplotlib axis object
        """
        if data != "I" and time not in ('dt',):
            raise AssertionError('(data, time) must equal (I, dt)')

        inf_data = self.ks

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if not test:
            for key, inf_test in inf_data.items():
                ax.plot(inf_test['ss_curve'][time], inf_test['ss_curve'][data],
                        label=r'$ss\ {}\ {}$'.format(data, key),
                        color=self.colors[key],
                        *args, **kwargs)
                ax.plot(inf_test['ss_data'][time][0],
                        inf_test['ss_data'][data][0],
                        # label=r'asymptote ${}$'.format(key),
                        color=self.colors[key],
                        marker='s')

        else:
            for key in test:
                try:
                    inf_test = inf_data[key]
                    ax.plot(inf_test['ss_curve'][time],
                            inf_test['ss_curve'][data],
                            label=r'$ss\ {}\ {}$'.format(data, key), *args,
                            **kwargs)
                    ax.plot(inf_test['ss_data'][time][0],
                            inf_test['ss_data'][data][0],
                            label=r'asymptote ${}$'.format(key),
                            color=self.colors[key],
                            marker='s')
                except KeyError:
                    pass

        # ax.legend(loc=0)
        ax.set_ylabel(r'$cm$', fontsize=20)

        ax.set_xlabel(r'$s$', fontsize=20)

        if savefig:
            plt.legend(loc=0, fontsize=12, numpoints=1)
            plt.savefig("Ks_ssi_{}.png".format(self.__file))
            plt.close()

        else:
            return ax

    def plot_unsaturated(self, data='gardner_curve', savefig=False,
                         *args, **kwargs):
        """
        User method to plot unsaturated Gardner curve kh, h

        Args:
            data:
            savefig:
            *args:
            **kwargs:

        Returns:
            matplotlib axis object
        """

        if data not in ("gardner_curve", "lf_advance"):
            raise NotImplementedError('please use gardner_curve for plotting')

        fig = plt.figure()
        ax = fig.add_subplot(111)

        inf_data = self.unsaturated

        if data == "gardner_curve":
            ax.plot(inf_data[data]['kh'], inf_data[data]['h'],
                    *args, **kwargs)

            ax.set_yscale('log')
            ax.set_ylim([10**5, 10**0])
            ax.set_ylabel(r'$h\ [-cm]$', fontsize=18)

            ax.set_xlabel(r'$K(h)$', fontsize=18)

        else:
            ax.plot(inf_data[data]['dt'], inf_data[data]['Lf'],
                    label='Green-Ampt', *args, **kwargs)
            ax.plot(inf_data['lf_advance2']['dt'],
                    inf_data['lf_advance2']['Lf'],
                    label="Unit Gradient")
            ax.set_ylim([np.max(inf_data[data]['Lf']), 0])
            ax.set_ylabel(r'$cm$', fontsize=18)
            ax.set_xlabel(r'$s$', fontsize=18)
            plt.legend(loc=0, fontsize=12)

        if savefig:
            plt.savefig("{}_{}.png".format(data, self.__file))
            plt.close()

        else:
            return ax

    def plot_filtering(self, data='depth', time='dt', test=(),
                       ax=None, savefig=False, *args, **kwargs):
        """
        User method to plot Savitzky-Golay filtered data and raw data
        on single grid.
        Args:
            data:
            time:
            test:
            ax:
            savefig:
            *args:
            **kwargs:

        Returns:
            ax: matplotlib axis object
        """

        if not self.filtered_infiltration:
            raise AssertionError('Data has not been filtered')


        filtered_inf_data = self.filtered_infiltration
        inf_data = self.raw_infiltration

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if not test:
            for key, inf_test in inf_data.items():
                mec = colorConverter.to_rgba(self.colors[key], alpha=0.6)
                ax.plot(inf_test[time], inf_test[data], 'o',
                        label=r'${}\ {}$'.format(data, key),
                        ms=8, mfc='None', mec=mec)
                ax.plot(filtered_inf_data[key][time],
                        filtered_inf_data[key][data],
                        label=r'$SG\ {}\ {}$'.format(data, key),
                        color=self.colors[key], lw=2)

        else:
            for key in test:
                try:
                    inf_test = inf_data[key]
                    mec = colorConverter.to_rgba(self.colors[key], alpha=0.6)
                    ax.plot(inf_test[time], inf_test[data],
                            label=r'${}\ {}$'.format(data, key),
                            ms='o', mfc='None', mec=mec)
                    ax.plot(filtered_inf_data[key][time],
                            filtered_inf_data[key][data],
                            label=r'$SG\ {}\ {}'.format(data, key),
                            color=self.colors[key])
                except KeyError:
                    pass

        ax.legend(loc=0, fontsize=12, numpoints=1)

        if data == 'i':
            ax.set_ylabel(r'$cm/s$', fontsize=20)
        else:
            ax.set_ylabel(r'$cm$', fontsize=20)

        if time == "sqrt_dt":
            ax.set_xlabel(r'$s^{1/2}$', fontsize=20)
        else:
            ax.set_xlabel(r'$s$', fontsize=20)

        if savefig:
            plt.savefig("{}_filtering_{}.png".format(data, self.__file))
            plt.close()

        else:
            return ax

    def create_tables(self):
        """
        User method to create matplotlib tables of processed data
        """
        if self.filtered_infiltration:
            inf_data = self.filtered_infiltration
        else:
            inf_data = self.raw_infiltration

        if not self.sorptivity:
            self.calculate_sorptivity()
        sorp_data = self.sorptivity

        if not self.ks:
            self.calculate_ks()
        ks_data = self.ks

        if not self.unsaturated:
            self.calculate_unsaturated()
        unsat_data = self.unsaturated

        header = (r'$Test$', r'$i\ cm/s$', r'$i\ r^{2}$',
                  r'$S\  cm/s^{1/2}$', r'$\beta$', r'$n_{eff}$', r'$K_s\ cm/s$')

        dm = []
        for key, inf_test in sorted(ks_data.items()):
            beta = self.__estimate_gardner_beta(n=inf_test['n'],
                                                ks=inf_test['ks'],
                                                b=inf_test['b'],
                                                s=sorp_data[key]['S'])
            dm.append((key, "{:.2e}".format(inf_test['ss_i']),
                       "{:.2e}".format(inf_test['r2']),
                       '{:.2e}'.format(sorp_data[key]['S']),
                       "{:.2e}".format(beta),
                       "{:.2e}".format(inf_test['n']),
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

        plt.savefig("{}_table1.png".format(self.__file))
        plt.close()

        header1 = (r'$\bar\beta$', r'$\bar n_{eff}$', r'$\sigma\ n_{eff}$',
                  r'$\bar i\ cm/s$', r'$\sigma\ i$', r'$\bar K_s\ cm/s$',
                  r'$\sigma\ K_s$', r'$K_{ss}\ cm/s$', r'$L_f\ cm$')

        dm1 = [('{:.2e}'.format(unsat_data['beta']),
                '{:.2e}'.format(unsat_data['n'][0]),
                '{:.2e}'.format(unsat_data['n'][1]),
                '{:.2e}'.format(unsat_data['ss_i'][0]),
                '{:.2e}'.format(unsat_data['ss_i'][1]),
                '{:.2e}'.format(unsat_data['ks'][0]),
                '{:.2e}'.format(unsat_data['ks'][1]),
                '{:.2e}'.format(inf_test['ks']),
                '{:.2e}'.format(unsat_data['wetting_front']))]

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
        plt.savefig("{}_table2.png".format(self.__file))

        with open("{}_analysis.txt".format(self.__file), 'w') as af:
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
    ptfile = 'Trod3.csv'
    fht = FallingHeadOnlyTests(ptfile, sensitivity=1.5, ignore_test=(2,))
    s = fht.calculate_sorptivity(sqrt_dt=3)

    ks = fht.calculate_ks(precision=0.003)
    uz = fht.calculate_unsaturated(mean_ks=True)
    fht.plot_infiltration(data='I', time='dt', savefig=True, linewidth=2)

    y = fht.plot_infiltration(data='I', time='sqrt_dt', linewidth=2)
    fht.plot_sorptivity(ax=y, savefig=True, ls='--')

    x = fht.plot_infiltration(data='I', time='dt', linewidth=2)
    fht.plot_ks(ax=x, savefig=True, ls='--', lw=2)

    fht.plot_filtering(savefig=True)
    fht.plot_unsaturated(savefig=True)
    fht.plot_unsaturated(data='lf_advance', savefig=True)
    fht.create_tables()

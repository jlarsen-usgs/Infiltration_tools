"""
__author__: Joshua D Larsen

Infiltration toolbox piece to analyze falling head infiltration data
from multiple falling head tests.

Can return estimates of sorptivity, hydraulic conductivity, effective porosity
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


class MariotteTest(object):
    """
    Class that seperates curves and builds out falling head infiltration test
    curves and tables

    Parameters:
        ir_file: (str) win_situ pressure transducer file name for inner ring
        mar_file: (str) win_situ pressure transducer file name for
            mariotte bottle
        ir_radius: (float) inner ring radius of infiltrometer in cm
        mar_radius: (float) inner radius of mariotte bottle in cm
        pt_unit: (str) base units of the pressure transducer
        sensitivity: (float) sensitivity to change in data to recognize
            beginning of test (cm)
        filter: (bool) flag to indicate whether to apply savgol golay filtering
            to raw data (recommended!)
        curve_sep: (int) seperation sensitivity default 300,
            increase value with noisy data if curves not seperating properly
        curve_shift: (int) shift seperation index, default is 0
            useful when data is noisy and we don't get proper curve separation
        ignore: (int) ignore the first x number of values due, default 0
            useful when mariotte pressure equalizing still and returns flat I
    """
    def __init__(self, ir_file, mar_file=None, pt_unit='ft', ir_radius=30.48,
                 mar_radius=23.01875, sensitivity=3.048, filter=True,
                 curve_sep=300, curve_shift=0, ignore=0):
        self.__irfile = ir_file
        self.__marfile = mar_file
        self.mar_radius = mar_radius
        self.ir_radius = ir_radius
        self.curve_separation = curve_sep
        self.curve_shift = curve_shift
        self.ignore = ignore
        self.data_startline = 0
        self.pt_unit = pt_unit
        self.time_index = 0
        self.depth_index = 0
        self.header = None
        self.raw_data = []
        self.raw_mariotte = []
        self.raw_infiltration = {}
        self.filtered_infiltration = {}
        self.sensitivity = sensitivity
        self.sorptivity = {}
        self.ks = {}
        self.unsaturated = {}
        self.colors = {'constant_head': "b", "falling_head": "g"}

        t = self.unit_to_cm

        self.__filter = filter
        self.__time_depth = []
        self.__mariotte_time_depth = []

        self.__get_raw_data()
        self.__get_time_depth_data()

        self.__get_raw_data(ftype='mariotte')
        self.__get_time_depth_data(ftype='mariotte')

        self.__get_inner_ring_test_curve()
        self.__seperate_constant_falling_curve()
        self.__get_mariotte_test_curve()
        self.__get_infiltration_curves()

        if self.__filter:
            self.__filter_data()
            self.__get_infiltration_curves(filtered=True)

    def calculate_sorptivity(self, sqrt_dt=3):
        """
        Use the early time Phillips equation solution to get an estimate
        of Sorptivity for infiltration tests.

        Parameters:
            sqrt_dt: (float) time cutoff for the calcuation of sorptivity

        Returns:

        """
        if self.__filter:
            inf_test = self.filtered_infiltration['constant_head']
        else:
            inf_test = self.raw_infiltration['constant_head']

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

        self.sorptivity['constant_head'] = {"S": S, "S_data": sorptivity_data,
                                            "S_curve": sorptivity_curve}

        return self.sorptivity

    def calculate_ks(self, precision=0.005):
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

        cnst_test = inf_data['constant_head']

        ks_data = np.recarray((1,), dtype=cnst_test.dtype)
        asymp_index, asymp_precision = self.__find_asymptote(cnst_test,
                                                             precision=precision)

        for idx, record in enumerate(cnst_test[asymp_index:]):
            if idx == 0:
                ks_data[0] = tuple(record)
            else:
                ks_data.resize((len(ks_data) + 1,),
                               refcheck=False)
                ks_data[-1] = tuple(record)

        regress = linregress(ks_data['dt'], ks_data['I'])
        ss_i = regress.slope
        ss_b = regress.intercept
        r_sq = regress.rvalue ** 2

        ss_i_curve = np.recarray((2,), dtype=[('dt', np.float),
                                              ('I', np.float)])

        ss_i_curve[0] = (cnst_test['dt'][0],
                         ss_i * cnst_test['dt'][0] + ss_b)

        ss_i_curve[-1] = (cnst_test['dt'][-1],
                          ss_i * cnst_test['dt'][-1] + ss_b)

        mean_h = np.mean(ks_data['depth'])
        sorptivity = self.sorptivity['constant_head']["S"]
        ks = self.__calculate_phillip_ks(cnst_test['I'],
                                         sorptivity,
                                         cnst_test['dt'],
                                         bnds=((1e-8, ss_i),))

        beta = self.__calculate_reynolds_beta(ks, ss_i, mean_h,
                                              self.ir_radius)
        n_eff = self.__estimate_eff_porosity(ks, beta, sorptivity)

        fall_test = inf_data['falling_head']

        regress = linregress(fall_test['dt'], fall_test['I'])
        fh_ss_i = regress.slope
        fh_ss_b = regress.intercept
        fh_r_sq = regress.rvalue ** 2

        fh_ss_i_curve = np.recarray((2,), dtype=[('dt', np.float),
                                                 ('I', np.float)])

        fh_ss_i_curve[0] = (fall_test['dt'][0],
                            fh_ss_i * fall_test['dt'][0] + fh_ss_b)

        fh_ss_i_curve[-1] = (fall_test['dt'][-1],
                             fh_ss_i * fall_test['dt'][-1] + fh_ss_b)


        f_ks = fh_ss_i * (3./2.)  # Based on phillips assumptions
        f_ks2 = self.__calculate_phillip_ks(fall_test["I"],
                                            sorptivity,
                                            fall_test['dt'],
                                            bnds=((1e-8, fh_ss_i),))

        self.ks['constant_head'] = {'ks': ks, 'ss_data': ks_data,
                                    'ss_curve': ss_i_curve, 'ss_i': ss_i,
                                    'n': n_eff, 'r2': r_sq, 'b': 0.55,
                                    'beta': beta, 'h': mean_h}
        self.ks['falling_head'] = {'ks_darcy': f_ks, 'ks': f_ks2,
                                   'ss_i': fh_ss_i, 'r2': fh_r_sq,
                                   'ss_curve': fh_ss_i_curve}

        return self.ks

    def calculate_unsaturated(self):
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

        ks_data = self.ks

        ks_cnst = ks_data['constant_head']
        ks_fall = ks_data['falling_head']

        cnst_test = inf_data['constant_head']
        fall_test = inf_data['falling_head']

        ks = [ks_cnst['ks'], ks_fall['ks'], ks_fall['ks_darcy']]
        ks_std = np.std(ks)
        ks = np.mean(ks)
        ss_i = [ks_cnst['ss_i'], ks_fall['ss_i']]
        i_std = np.std(ss_i)
        ss_i = np.mean(ss_i)
        n = ks_cnst['n']
        beta = ks_cnst['beta']
        sorptivity = self.sorptivity['constant_head']['S']
        mean_h = ks_cnst['h']

        kh_curve = self.__estimate_kh(ks, beta)

        cnst_lf = self.__estimate_wetting_front_depth(10., mean_h,
                                                      ks,
                                                      cnst_test['dt'][-1], n)

        wf_advance = np.recarray((1,), dtype=[('dt', np.float),
                                              ('delh', np.float),
                                              ('Lf', np.float)])

        for idx, record in enumerate(cnst_test):
            if idx == 0:
                lf = self.__estimate_wetting_front_depth(0.1, record['depth'],
                                                         ks_cnst['ks'],
                                                         1., n)

                wf_advance[0] = (record['dt'], record['depth'], lf)
            else:
                lf = self.__estimate_wetting_front_depth(wf_advance['Lf'][-1],
                                                         record['depth'],
                                                         ks,
                                                         idx, n)

                wf_advance.resize((len(wf_advance) + 1,), refcheck=False)
                wf_advance[-1] = (record['dt'], record['depth'], lf)

        cnst_time = idx + 1

        for idx, record in enumerate(fall_test):
            lf = self.__estimate_wetting_front_depth(wf_advance['Lf'][-1],
                                                     record['depth'],
                                                     ks,
                                                     cnst_time + idx, n)

            wf_advance.resize((len(wf_advance) + 1,), refcheck=False)
            wf_advance[-1] = (record['dt'], record['depth'], lf)

        lf_cnst = cnst_test['I']/n
        lf_fall = fall_test['I']/n

        wf_advance2 = np.recarray((len(lf_cnst) + len(lf_fall),),
                                  dtype=[('dt', np.float),('Lf', np.float)])
        for idx, record in enumerate(lf_cnst):
            wf_advance2[idx] = (idx, record)
            wf_idx = idx + 1

        for idx, record in enumerate(lf_fall):
            wf_advance2[wf_idx + idx] = (wf_idx + idx, record)

        self.unsaturated = {'beta': beta, 'gardner_curve': kh_curve,
                            'wetting_front': wf_advance2[-1]['Lf'], 'n': n,
                            'ks': (ks, ks_std), 'ss_i': (ss_i, i_std),
                            'lf_advance': wf_advance,
                            'lf_advance2': wf_advance2}

    def __estimate_wetting_front_depth(self, lf, delh, ks, t, n):
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
        f = lf - delh * np.log(1 + lf / delh) - (ks * t) / n
        f_prime = lf / (delh + lf)

        lf1 = lf - (f / f_prime)

        if abs(lf1 - lf) > 0.01:
            lf1 = self.__estimate_wetting_front_depth(lf1, delh, ks, t, n)

        return lf1

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

    def __estimate_eff_porosity(self, ks, beta, sorptivity):
        """
        Estimate effective porosity based upon White and Sulley 1987

        Parameters:
            beta: (float) gardners beta
            sorptivity: (float) sorptivity

        Returns:
            n_eff: (float) effective porosity
        """
        return (beta * np.pi/4 * sorptivity**2)/ks

    def __calculate_reynolds_beta(self, ks, ssi, h, a_r, d=3.048,
                                bnds=((1e-6, 1.),)):
        """
        Method to apply Reynolds et. al. 2002  to calculate beta from
        steady state infiltration rate. Returns Gardners Beta

        Parameters:
            ks: (float) saturated hydraulic conductivity
            ssi: (float) steady state infiltration rate
            h: (float) steady state constant head
            a_r: (float) inner ring radius
            d: (float) depth of ring insertion

        Returns:
            beta
        """
        x = minimize(self.__reynolds_ks_residual, [0.001],
                     args=(ks, ssi, h, a_r, d))

        return 1/x['x'][0]

    def __reynolds_ks_residual(self, x, ks, ssi, h, a_r, d):
        """
        Minimization method for Reynolds 2002. Solves for the inverse
        Gardners Beta
        Parameters:
            x: (list) initial guess for ks, beta
            ssi: (float) steady state infiltration rate
            h: (float) steady state constant head
            a_r: (float) inner ring radius
            d: (float) depth of ring insertion

        Returns:
            residual
        """

        r0 = h / (0.316 * np.pi * d + 0.184 * np.pi * a_r)
        r1 = 1. / (x[0] * (0.316 * np.pi * d + 0.184 * np.pi * a_r))
        return np.abs(ks*(r0 + r1 + 1) - ssi)

    def __calculate_phillip_ks(self, I, sorptivity, dt, bnds):
        """
        Method to apply Phillip 1957 to cumulative infiltration data
        to estimate Ks from I = At + St^(1/2) where A = 2/3 Ks via Youngs 1968

        Parameters:
            I: (np.array) cumulative (steady state) infiltration
            sorptivity: (float) sorptivity
            dt: (float) delta time array
        returns:
            ks: (float) 1d saturated hydraulic conductivity
        """
        I = np.array(list(I))
        dt = np.array(list(dt))
        a = minimize(self.__phillip_ks_residual, [1.0],
                     args=(I, sorptivity, dt),
                     bounds=bnds, method="SLSQP")
        return a['x'][0] * (3./2.)  # multiplication by 3./2. via Youngs 1968


    def __phillip_ks_residual(self, a, I, sorptivity, dt):
        """
        method to solve for phillip ks via residual using minimization

        Parameters:
            a: (np.array, float) phillip's A term
            I: (np.array) cumulative (steady state) infiltration
            sorptivity: (float) sorptivity
            dt: (float) delta time array
        Returns:
            residual
        """
        return np.abs(np.sum(I - (a[0] * dt - sorptivity * np.sqrt(dt))))

    def __filter_data(self):
        """
        Method applies a 9 point Savgol Golay filter to infiltration data
        and sets data to self.filtered_infiltration
        """
        for key, inf_test in self.raw_infiltration.items():
            if key == 'constant_head':
                filtered_inf = np.recarray((len(inf_test),),
                    dtype=np.dtype([('time', np.object), ('dt', np.float),
                                    ('sqrt_dt', np.float), ('depth', np.float),
                                    ('mariotte_depth', np.float)]))

                depth = np.array([i for i in inf_test['depth']])
                filtered_depth = savgol_filter(depth, 9, 1)

                mar_depth = np.array([i for i in inf_test['mariotte_depth']])
                filtered_mariotte = savgol_filter(mar_depth, 9, 1)


                for idx, rec in enumerate(inf_test):
                    filtered_inf[idx] = (rec['time'], rec['dt'],
                                         rec['sqrt_dt'], filtered_depth[idx],
                                         filtered_mariotte[idx])

            else:
                filtered_inf = np.recarray((len(inf_test),),
                    dtype=np.dtype([('time', np.object), ('dt', np.float),
                                    ('sqrt_dt', np.float),
                                    ('depth', np.float)]))

                depth = np.array([i for i in inf_test['depth']])
                filtered_depth = savgol_filter(depth, 9, 1)

                for idx, rec in enumerate(inf_test):
                    filtered_inf[idx] = (rec['time'], rec['dt'], rec['sqrt_dt'],
                                         filtered_depth[idx])

            self.filtered_infiltration[key] = filtered_inf

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
                                                         precision=precision + 0.001)
        return asymp_idx, precision

    def __get_infiltration_curves(self, filtered=False):
        """
        Method to calculate cumulative infiltration and infiltration rate
        curves for both falling and constant head tests.

        sets results to self.raw_infiltration
        """
        if filtered:
            cnst_test = self.filtered_infiltration['constant_head']
            fall_test = self.filtered_infiltration['falling_head']
        else:
            cnst_test = self.raw_infiltration['constant_head']
            fall_test = self.raw_infiltration['falling_head']

        constant_head = np.recarray((len(cnst_test),),
                                    dtype=[('time', np.object),
                                           ('dt', np.float),
                                           ('sqrt_dt', np.float),
                                           ('depth', np.float),
                                           ('mariotte_depth', np.float),
                                           ('I', np.float),
                                           ('i', np.float)])

        for idx, record in enumerate(cnst_test):
            if idx == 0:
                I = 0
                i = 0
            else:
                I0 = constant_head["I"][idx - 1]
                delta_d_ir = constant_head["depth"][idx - 1] - record['depth']

                delta_d_mb = constant_head["mariotte_depth"][idx - 1] - \
                    record['mariotte_depth']

                # todo: figure out how to fix this loss equation or how to
                # todo: apply ignore somewhere like when we grab the initial curve

                I = I0 + (delta_d_ir +
                          (delta_d_mb * (self.mar_radius/self.ir_radius)**2))

                i = I/record['dt']

                if i <= 0:
                    # use this as a repair for when mariotte is still pressurizing

                    I = I0 + (delta_d_ir -
                              (delta_d_mb * (
                              self.mar_radius / self.ir_radius) ** 2))
                    i = I/record['dt']
                    #else:
                    #    I = I0
                    #    i = constant_head[idx - 1]['i']

            constant_head[idx] = (record['time'], record['dt'],
                                  record['sqrt_dt'], record['depth'],
                                  record['mariotte_depth'], I, i)

        falling_head = np.recarray((len(fall_test),),
                                   dtype=[('time', np.object),
                                          ('dt', np.float),
                                          ('sqrt_dt', np.float),
                                          ('depth', np.float),
                                          ('I', np.float),
                                          ('i', np.float)])

        I0 = constant_head["I"][-1]
        d0 = constant_head['depth'][-1]
        for idx, record in enumerate(fall_test):
            I = I0 + (d0 - record['depth'])
            i = I/record['dt']

            falling_head[idx] = (record['time'], record['dt'],
                                 record['sqrt_dt'], record['depth'], I, i)

        if filtered:
            self.filtered_infiltration['constant_head'] = constant_head
            self.filtered_infiltration['falling_head'] = falling_head
        else:
            self.raw_infiltration['constant_head'] = constant_head
            self.raw_infiltration['falling_head'] = falling_head

    def __seperate_constant_falling_curve(self):
        """
        Method that iterates over the data creating two regression lines
        to find the inflection point based upon the minimum angle of
        intersection of those regression lines.

        seperates out constant and falling test based on inflection point
        and sets these to self.raw_infiltration
        """

        inf_curve = self.raw_infiltration['full_test']

        d0 = inf_curve['depth'][0]
        t0 = inf_curve['dt'][0]
        d1 = inf_curve['depth'][-1]
        t1 = inf_curve['dt'][-1]

        sep = self.curve_separation
        shift = self.curve_shift
        # get data of interest to draw regression lines
        inf_curve = inf_curve[sep:-sep]
        min_angle = 180
        inflection_idx = 0

        for idx, record in enumerate(inf_curve):
            s0 = (record['depth'] - d0)/float(record['dt'] - t0)
            s1 = (record['depth'] - d1)/float(record['dt'] - t1)

            angle = np.pi - abs(np.pi + np.arctan(s0) - np.arctan(s1))
            angle = np.rad2deg(angle)
            if angle < min_angle:
                inflection_idx = idx + sep - shift
                min_angle = angle

        self.raw_infiltration['constant_head'] =\
            self.raw_infiltration['full_test'][:inflection_idx]

        self.raw_infiltration['falling_head'] =\
            self.raw_infiltration['full_test'][inflection_idx:]

    def __get_inner_ring_test_curve(self):
        """
        Method to grab the full inner ring test curve from the complete set
        of win-situ data. This data is set to self.raw_infiltration['full_test']
        """
        curves = {}
        test_begins = False
        logging_test = False
        test_no = "full_test"
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
                        if (line[-1] - self.__time_depth[idx + 10][-1]) > 0:
                            test = [self.__time_depth[idx - 1], line]
                            logging_test = True
                            test_begins = False

                elif logging_test:
                    test.append(line)
                    if line[-1] <= 0.:
                        logging_test = False
                        curves[test_no] = test
                        break

        if logging_test:
            curves[test_no] = test

        for key, inf_test in curves.items():
            if self.ignore:
                inf_test = inf_test[self.ignore:]
            infiltration = np.recarray((len(inf_test),),
                                       dtype=np.dtype([('time', np.object),
                                                       ('dt', np.float),
                                                       ('sqrt_dt', np.float),
                                                       ('depth', np.float)]))
            t0, d0 = inf_test[0]
            for idx, inf_rec in enumerate(inf_test):
                delt = dt.datetime.combine(dt.date.min, inf_rec[0]) -\
                       dt.datetime.combine(dt.date.min, t0)
                delt = delt.seconds
                sqrt_time = np.sqrt(delt)
                infiltration[idx] = (inf_rec[0], delt, sqrt_time,
                                     inf_rec[-1])

            self.raw_infiltration[key] = infiltration

    def __get_mariotte_test_curve(self):
        """
        Method to get the mariottte bottle test curve by matching data
        from the seperated inner ring curve
        """
        inf_data = self.raw_infiltration
        mar_data = self.__mariotte_time_depth

        cnst_test = inf_data['constant_head']

        align_idx = 0
        for idx, record in enumerate(mar_data):
            if record[0] == cnst_test['time'][0]:
                align_idx = idx
                break

        mar_data = mar_data[align_idx:]

        constant_head = np.recarray((len(cnst_test),),
                                    dtype=[('time', np.object),
                                           ('dt', np.float),
                                           ('sqrt_dt', np.float),
                                           ('depth', np.float),
                                           ('mariotte_depth', np.float)])

        for idx, record in enumerate(cnst_test):
            try:
                mar_depth = mar_data[idx][-1]
            except IndexError:
                mar_depth = 0.0000

            constant_head[idx] = (record['time'], record['dt'],
                                  record['sqrt_dt'], record['depth'],
                                  mar_depth)


        self.raw_infiltration['constant_head'] = constant_head

    def __get_time_depth_data(self, ftype='inner_ring'):
        """
        Background method that grabs the time object and depth of ponding
        """
        time_depth = []
        if ftype == 'mariotte':
            raw_data = self.raw_mariotte
            mariotte = True
        else:
            raw_data = self.raw_data
            mariotte = False

        for record in raw_data:
            try:
                time = self.__format_time(record[self.time_index])
                depth = float(record[self.depth_index]) * self.unit_to_cm
                if mariotte:
                    time_depth.append([time, -1 * depth])
                else:
                    time_depth.append([time, depth])
            except ValueError:
                pass

        if mariotte:
            self.__mariotte_time_depth = time_depth
        else:
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
                return dt.time(hour=int(t[0])+12,
                               minute=int(t[1]),
                               second=int(t[2]))

        return dt.time(hour=int(t[0]), minute=int(t[1]), second=int(t[2]))

    def __get_raw_data(self, ftype='inner_ring'):
        """
        Method to analyze the csv file and grab only the pressure transducer
        data from it
        """
        if ftype == "mariotte":
            fname = self.__marfile
            mariotte = True
        else:
            fname = self.__irfile
            mariotte = False

        self.__get_data_startline(fname)
        raw_data = []

        with open(fname) as ptf:
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
        if mariotte:
            self.raw_mariotte = raw_data
        else:
            self.raw_data = raw_data

    def __get_data_startline(self, fname):
        """
        Method to find which line pt data begins in win situ csv files
        """
        data0 = False
        data1 = False
        with open(fname) as ptf:
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
            for key in ('constant_head', 'falling_head'):
                inf_test = inf_data[key]
                labels = key.split('_')
                ax.plot(inf_test[time], inf_test[data],
                        label=r'${}\ {}\ {}$'.format(data, labels[0], labels[1]),
                        color=self.colors[key],
                        *args, **kwargs)

        else:
            for key in test:
                try:
                    inf_test = inf_data[key]
                    labels = key.split('_')
                    ax.plot(inf_test[time], inf_test[data],
                            label=r'${}\ {}\ {}$'.format(data, labels[0], labels[1]),
                            *args, **kwargs)
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
            plt.savefig("{}_{}.png".format(data, self.__irfile))
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

        inf_test = self.sorptivity['constant_head']

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)


        ax.plot(inf_test['S_curve'][time], inf_test['S_curve'][data],
                label=r'$S$', color=self.colors['constant_head'],
                *args, **kwargs)


        ax.set_ylabel(r'$cm$', fontsize=20)
        ax.set_xlabel(r'$s^{1/2}$', fontsize=20)

        if savefig:
            plt.legend(loc=0, fontsize=12)
            plt.savefig("S_{}.png".format(self.__irfile))
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
            for key in ('constant_head', 'falling_head'):
                labels = key.split('_')
                ax.plot(inf_data[key]['ss_curve'][time],
                        inf_data[key]['ss_curve'][data],
                        label=r'$ss\ {}\ {}\ {}$'.format(data, labels[0],
                                                         labels[1]),
                        color=self.colors[key],
                        *args, **kwargs)
                if key == 'constant_head':
                    ax.plot(inf_data[key]['ss_data'][time][0],
                            inf_data[key]['ss_data'][data][0],
                            # label=r'asymptote ${}$'.format(key),
                            color=self.colors[key],
                            marker='s')

        else:
            for key in test:
                try:
                    inf_test = inf_data[key]
                    labels = key.split('_')
                    ax.plot(inf_test['ss_curve'][time],
                            inf_test['ss_curve'][data],
                            label=r'$ss\ {}\ {}\ {}$'.format(data, labels[0],
                                                             labels[1]),
                            **kwargs)
                    if key == 'contant_head':
                        ax.plot(inf_test['ss_data'][time][0],
                                inf_test['ss_data'][data][0],
                                # label=r'asymptote ${}$'.format(key),
                                color=self.colors[key],
                                marker='s')
                except KeyError:
                    pass

        # ax.legend(loc=0)
        ax.set_ylabel(r'$cm$', fontsize=20)

        ax.set_xlabel(r'$s$', fontsize=20)

        if savefig:
            plt.legend(loc=0, fontsize=12, numpoints=1)
            plt.savefig("Ks_ssi_{}.png".format(self.__irfile))
            plt.close()

        else:
            return ax

    def plot_unsaturated(self, data='gardner_curve', savefig=False,
                         *args, **kwargs):
        """
        User method to plot unsaturated Gardner curve kh, h and modeled
        wetting front

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
            ax.set_ylim([10 ** 5, 10 ** 0])
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
            plt.savefig("{}_{}.png".format(data, self.__irfile))
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
        colors = {'constant_head': 'k', 'falling_head': 'grey'}

        if not test:
            for key in ('constant_head', 'falling_head'):
                inf_test = inf_data[key]
                labels = key.split('_')
                mec = colorConverter.to_rgba(self.colors[key], alpha=0.6)
                ax.plot(inf_test[time], inf_test[data], 'o',
                        label=r'${}\ {}\ {}$'.format(data,
                                                     labels[0], labels[1]),
                        ms=8, mfc='None', mec=mec)
                ax.plot(filtered_inf_data[key][time],
                        filtered_inf_data[key][data],
                        label=r'$SG\ {}\ {}\ {}$'.format(data,
                                                         labels[0], labels[1]),
                        color=colors[key], lw=2)

        else:
            for key in test:
                try:
                    inf_test = inf_data[key]
                    labels = key.split('_')
                    mec = colorConverter.to_rgba(self.colors[key], alpha=0.6)
                    ax.plot(inf_test[time], inf_test[data],
                            label=r'${}\ {}\ {}$'.format(data,
                                                         labels[0], labels[1]),
                            ms='o', mfc='None', mec=mec)
                    ax.plot(filtered_inf_data[key][time],
                            filtered_inf_data[key][data],
                            label=r'$SG\ {}\ {}\ {}$'.format(data,
                                                             labels[0],
                                                             labels[1]),
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
            plt.savefig("{}_filtering_{}.png".format(data, self.__irfile))
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
        for key in ('constant_head', 'falling_head'):
            inf_test = ks_data[key]
            if key == 'constant_head':
                dm.append((key.split("_")[0],
                           "{:.2e}".format(inf_test['ss_i']),
                           "{:.2e}".format(inf_test['r2']),
                           '{:.2e}'.format(sorp_data["constant_head"]['S']),
                           "{:.2e}".format(inf_test['beta']),
                           "{:.2e}".format(ks_data['constant_head']['n']),
                           "{:.2e}".format(inf_test['ks'])))
            else:
                dm.append((key.split("_")[0],
                           "{:.2e}".format(inf_test['ss_i']),
                           "{:.2e}".format(inf_test['r2']),
                           '---',
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

        plt.savefig("{}_table1.png".format(self.__irfile))
        plt.close()

        header1 = (r'$\bar\beta$', r'$\bar n_{eff}$', r'$\sigma\ n_{eff}$',
                   r'$\bar i\ cm/s$', r'$\sigma\ i$', r'$\bar K_s\ cm/s$',
                   r'$\sigma\ K_s$', r'$K_{ss}\ cm/s$', r'$L_f\ cm$')

        dm1 = [('{:.2e}'.format(unsat_data['beta']),
                '{:.2e}'.format(unsat_data['n']),
                '---',
                '{:.2e}'.format(unsat_data['ss_i'][0]),
                '{:.2e}'.format(unsat_data['ss_i'][1]),
                '{:.2e}'.format(unsat_data['ks'][0]),
                '{:.2e}'.format(unsat_data['ks'][1]),
                '{:.2e}'.format(inf_test['ks_darcy']),
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

        plt.savefig("{}_table2.png".format(self.__irfile))

        with open("{}_analysis.txt".format(self.__irfile), 'w') as af:
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


if __name__ == "__main__":
    irfile = "Inf11.csv"
    mbfile = "Inf11_mb.csv"
    mbt = MariotteTest(irfile, mbfile, sensitivity=1.5, ignore=50)#,
                       #ignore=100, curve_shift=-40)
    sdt = mbt.calculate_sorptivity(sqrt_dt=3)
    mks = mbt.calculate_ks(precision=0.001)
    unsat = mbt.calculate_unsaturated()

    #mbt.plot_infiltration(data='I', time='dt')
    #plt.show()

    mbt.plot_infiltration(data='i', time='dt', savefig=True)

    y = mbt.plot_infiltration(data='I', time='sqrt_dt')
    mbt.plot_sorptivity(ax=y, savefig=True, ls='--')

    x = mbt.plot_infiltration(data='I', time='dt')
    mbt.plot_ks(ax=x, savefig=True, ls='--', lw=2)

    # mbt.plot_unsaturated(savefig=True)
    mbt.plot_unsaturated(data='lf_advance', savefig=True)
    mbt.plot_filtering(savefig=True)

    mbt.create_tables()
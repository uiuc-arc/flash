import ast
import traceback as tb

import numpy as np
import pandas as pd
import scipy.stats.distributions as dist
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF

from src import Util
from src.lib.AssertType import AssertType

default_threshold = {AssertType.ASSERT_APPROX_EQUAL: {'significant': 10 ** -7},
                     AssertType.ASSERT_ALMOST_EQUAL: {'decimal': 1.5 * 10 ** (-7)},
                     AssertType.ASSERT_ALLCLOSE: {'rtol': 10 ** (-7), 'atol': 10 ** (-8)},
                     AssertType.ASSERT_ARRAY_ALMOST_EQUAL: {'decimal': 1.5 * 10 ** (-6)},
                     AssertType.TF_ASSERT_ALL_CLOSE: {'rtol': 1e-6},
                     AssertType.PYRO_ASSERT_EQUAL: {'prec': 1e-5}
                     }


class CompareDistribution:
    def __init__(self, samplesfile, actual, expected, assert_type, assert_str, dist_type=norm, logger=None):
        self.sample_file = samplesfile
        self.assert_type = assert_type
        self.assert_str = assert_str
        self.actual = actual
        self.expected = expected
        self.logger = logger
        self.enable_per_dim_comparison = True
        # can be either norm (Normal distribution) or ECDF (Empirical)
        self.dist_type = dist_type
        self.minimum_variance_threshold = 1e-20
        if samplesfile is not None:
            self.parse()
        else:
            self.parse_arguments(self.assert_str[self.assert_str.index("(") + 1:-1].strip().split(","))

    def parse_arguments(self, args):
        for arg in args:
            if 'decimal' in arg or 'significant' in arg:
                self.logger.logo("updating...")
                d = int(arg.split('=')[1].strip())
                if self.assert_type == AssertType.ASSERT_APPROX_EQUAL:
                    default_threshold[self.assert_type][arg.split("=")[0].strip()] = 10 ** -(d - 1)
                elif self.assert_type == AssertType.ASSERT_ALMOST_EQUAL:
                    default_threshold[self.assert_type][arg.split("=")[0].strip()] = 1.5 * 10 ** (-d)
                elif self.assert_type == AssertType.ASSERT_ARRAY_ALMOST_EQUAL:
                    default_threshold[self.assert_type][arg.split("=")[0].strip()] = 1.5 * 10 ** (-d)
                else:
                    self.logger.logo("unrecognized assert %s " % self.assert_str)
                    raise RuntimeError
            elif 'rtol' in arg:
                self.logger.logo("updating...")
                rtol = float(arg.split('=')[1].strip())
                if self.assert_type == AssertType.ASSERT_ALLCLOSE:
                    default_threshold[self.assert_type]['rtol'] = rtol
                elif self.assert_type == AssertType.TF_ASSERT_ALL_CLOSE:
                    default_threshold[self.assert_type]['rtol'] = rtol
                else:
                    self.logger.logo("unrecognized assert %s " % self.assert_str)
                    raise RuntimeError
            elif 'atol' in arg:
                self.logger.logo('updating...')
                atol = float(arg.split('=')[1].strip())
                if self.assert_type == AssertType.ASSERT_ALLCLOSE:
                    default_threshold[self.assert_type]['atol'] = atol
                else:
                    self.logger.logo("unrecognized assert %s " % self.assert_str)
                    raise RuntimeError
            elif 'prec' in arg:
                self.logger.logo("updating...")
                rtol = float(arg.split('=')[1].strip())
                if self.assert_type == AssertType.PYRO_ASSERT_EQUAL:
                    default_threshold[self.assert_type]['prec'] = rtol
                else:
                    self.logger.logo("unrecognized assert %s " % self.assert_str)
                    raise RuntimeError
            elif 'atol' in arg:
                self.logger.logo("ignoring atol")

    def compute_fit_score(self, samples):
        # https://nedyoxall.github.io/fitting_all_of_scipys_distributions.html
        # removing very large samples

        class DeltaDist:
            def __init__(self, vals):
                self.val = np.max(vals)

            def cdf(self, samples):
                if isinstance(samples, (list, np.ndarray)):
                    return [1.0 if self.val <= k else 0.0 for k in samples]
                else:
                    return 1.0 if self.val <= samples else 0.0

        try:
            # check if variance is too small, send the delta distribution
            var = np.var(np.hstack(samples))
            if var < self.minimum_variance_threshold:
                self.logger.logo("Variance ({0}) too small, using delta distribution".format(var))
                return "delta", DeltaDist(samples)
        except:
            self.logger.logo(tb.format_exc())
            self.logger.logo("Cannot compute variance, continuing with distribution computation...")

        models = [dist.norm, dist.expon, dist.gamma, dist.pareto, dist.t, dist.lognorm, dist.cauchy, dist.dweibull,
                  dist.invweibull,
                  dist.logistic, dist.beta, dist.gumbel_l, dist.halfcauchy, dist.laplace]

        # Returns un-normalised (i.e. counts) histogram
        y, x = np.histogram(np.array(samples), bins='sturges')

        # Some details about the histogram
        bin_width = x[1] - x[0]
        N = len(samples)
        x_mid = (x + np.roll(x, -1))[:-1] / 2.0  # go from bin edges to bin middles

        # selection of available distributions
        # CHANGE THIS IF REQUIRED
        DISTRIBUTIONS = models
        # print("Variance: {0}".format(np.var(samples)))
        # loop through the distributions and store the sum of squared errors
        # so we know which one eventually will have the best fit

        sses = []
        logpdfs = []
        for d in DISTRIBUTIONS:
            name = d.__class__.__name__[:-4]

            params = d.fit(np.array(samples))
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]

            pdf = d.pdf(x_mid, loc=loc, scale=scale, *arg)
            logpdf = d.logpdf(np.array(samples), loc=loc, scale=scale, *arg)
            pdf_scaled = pdf * bin_width * N  # to go from pdf back to counts need to un-normalise the pdf

            sse = np.sum((y - pdf_scaled) ** 2)
            sses.append([sse, name, d(loc=loc, scale=scale, *arg)])
            logpdfs.append([np.sum(logpdf), name, d(loc=loc, scale=scale, *arg)])

        # Things to return - df of SSE and distribution name, the best distribution and its parameters
        results = pd.DataFrame(sses, columns=['SSE', 'distribution', 'dist']).sort_values(by='SSE')
        best_name = results[np.isfinite(results['SSE'])].iloc[0]['distribution']
        best_dist = results[np.isfinite(results['SSE'])].iloc[0]['dist']
        # results_lpdf = pd.DataFrame(logpdfs, columns=['logpdf', 'distribution', 'dist']).sort_values(by='logpdf')
        # best_name = results_lpdf[np.isfinite(results_lpdf['logpdf'])].iloc[-1]['distribution']
        # best_dist = results_lpdf[np.isfinite(results_lpdf['logpdf'])].iloc[-1]['dist']

        self.logger.logo("Choosing distribution: {0}".format(best_name))
        return best_name, best_dist

    def parse(self):
        with open(self.sample_file, 'r') as f:
            lines = f.readlines()
            self.assert_type = AssertType.get_assert_type(lines[6].split(' ')[1].split('(')[0].split('.')[-1])
            self.assert_str = lines[6]
            self.logger.logo(self.assert_str.strip())
            args = lines[7].split(':')[1].strip()[1:-2].split(',')
            self.parse_arguments(args)

    def evaluate(self):
        if self.sample_file is not None:
            with open(self.sample_file, 'r') as sf:
                lines = sf.readlines()[9:]
                actual = [ast.literal_eval(x.split("::")[0].strip()) for x in lines]
                expected = [ast.literal_eval(x.split("::")[1].strip()) for x in lines]
        else:
            actual = np.array(self.actual)
            expected = np.array(self.expected)

        if len(actual) == 0:
            self.logger.logo("No Samples!!")
            return 1.0

        if self.assert_type in [AssertType.ASSERTLESS, AssertType.ASSERTLESSEQUAL]:
            p = self.check_assert_less(actual, expected)
        elif self.assert_type in [AssertType.ASSERTGREATER, AssertType.ASSERTGREATEREQUAL]:
            p = self.check_assert_greater(actual, expected)
        elif self.assert_type in [AssertType.ASSERT_ALLCLOSE]:
            p = self.check_assert_all_close_tolerance(actual, expected, default_threshold.get(self.assert_type))
        elif self.assert_type in [AssertType.PYRO_ASSERT_EQUAL, AssertType.ASSERT_ALMOST_EQUAL,
                                  AssertType.ASSERT_APPROX_EQUAL, AssertType.ASSERT_ARRAY_ALMOST_EQUAL]:
            p = self.check_assert_tolerance(actual, expected, default_threshold.get(self.assert_type))
        elif self.assert_type in [AssertType.ASSERT_ARRAY_LESS]:
            p = self.check_assert_less(actual, expected)
        elif self.assert_type in [AssertType.ASSERTTRUE, AssertType.ASSERTFALSE, AssertType.ASSERT]:
            if '<' in self.assert_str:
                p = self.check_assert_less(actual, expected)
            else:
                p = self.check_assert_greater(actual, expected)
        else:
            self.logger.logo("Unhandled assert %s " % self.assert_str)
            return 1.0

        if p > 0.0:
            self.logger.logo("Probability of fail (non-zero): %s" % str(p))
        else:
            self.logger.logo("Probability of fail : %s" % str(p))
        return p

    def check_assert_greater(self, actual: list, expected: list):
        if Util.getdims(actual) > 2:
            self.logger.logo("Higher dimension! Cannot handle")
            return 1.0
        if self.enable_per_dim_comparison and Util.getdims(actual) == 2:
            # for each index of actual, figure out the probability
            probs = []
            for ind in range(len(actual[0])):
                p = self.check_assert_greater([x[ind] for x in actual],
                                              expected if Util.getdims(expected) == 1 else [t[ind] for t in expected])
                probs.append(p)
            return np.max(probs)  # max probability of failing

        if self.dist_type is None:
            try:
                name, dist = self.compute_fit_score(actual)
                prob = dist.cdf(expected[0])
            except:
                self.logger.logo(tb.format_exc())
                return 1.0
        elif self.dist_type == norm:
            mean, var = norm.fit(actual)
            if var <= 0.0:
                var = 1e-20
            prob = norm.cdf(expected[0], loc=mean, scale=var)
        else:
            ecdf = ECDF(actual)
            prob = ecdf([expected[0]])[0]
        return np.max(prob)

    def check_assert_less(self, actual, expected):
        if Util.getdims(actual) > 2:
            self.logger.logo("Higher dimension! Cannot handle")
            return
        if self.enable_per_dim_comparison and (Util.getdims(actual) == 2 or isinstance(actual[0], (list, np.ndarray))):
            # for each index of actual, figure out the probability
            probs = []
            for ind in range(len(actual[0])):
                p = self.check_assert_less(np.array([arr.flatten()[ind] for arr in actual if ind < len(arr)]),
                                           expected if Util.getdims(expected) == 1 else
                                           np.array([arr.flatten()[ind] for arr in expected if ind < len(arr)]))
                probs.append(p)
            return np.max(probs)  # max probability of failing
        if self.dist_type is None:
            try:
                name, dist = self.compute_fit_score(actual)
                prob = dist.cdf(expected[0])
            except:
                import traceback as tb
                self.logger.logo(tb.format_exc())
                return 1.0
        elif self.dist_type == norm:
            mean, var = norm.fit(actual)
            if var <= 0.0:
                var = 1e-20
            prob = norm.cdf(expected[0], loc=mean, scale=var)
        else:
            ecdf = ECDF(actual)
            prob = ecdf([expected[0]])[0]
        return 1 - np.min(prob)  # ideally np.max(1-prob)

    # both absolute and relative
    def check_assert_all_close_tolerance(self, actual: list, expected: list, tol_thresh):
        if self.enable_per_dim_comparison and Util.getdims(actual) >= 2:
            # for each index of actual, figure out the probability
            probs = []
            for ind in range(len(actual[0])):
                p = self.check_assert_all_close_tolerance([x[ind] for x in actual if ind < len(x)],
                                                          expected if Util.getdims(expected) == 1 else [t[ind] for t in
                                                                                                        expected if
                                                                                                        ind < len(t)],
                                                          tol_thresh)
                probs.append(p)
            return np.max(probs)  # max probability of failing

        if self.dist_type is None:
            try:
                name, dist = self.compute_fit_score(np.subtract(np.abs(np.subtract(actual, expected)),
                                                                tol_thresh['atol']))
                prob = dist.cdf(tol_thresh['rtol'] * np.abs(expected))
            except:
                import traceback as tb
                self.logger.logo(tb.format_exc())
                return 1.0
        return 1 - np.min(prob)

    # relative tolerance
    def check_assert_relative_tolerance(self, actual: list, expected: list, tol_thresh):
        if isinstance(tol_thresh, dict):
            if 'rtol' in tol_thresh:
                tol_thresh = tol_thresh['rtol']
            elif 'decimal' in tol_thresh:
                tol_thresh = tol_thresh['decimal']
            elif 'significant' in tol_thresh:
                tol_thresh = tol_thresh['significant']

        if self.enable_per_dim_comparison and Util.getdims(actual) >= 2:
            # for each index of actual, figure out the probability
            probs = []
            for ind in range(len(actual[0])):
                p = self.check_assert_relative_tolerance([x[ind] for x in actual if ind < len(x)],
                                                         expected if Util.getdims(expected) == 1 else [t[ind] for t in
                                                                                                       expected if
                                                                                                       ind < len(t)],
                                                         tol_thresh)
                probs.append(p)
            return np.max(probs)  # max probability of failing
        if self.dist_type is None:
            try:
                name, dist = self.compute_fit_score(np.abs(np.subtract(actual, expected)))
                prob = dist.cdf(tol_thresh)
            except:
                import traceback as tb
                self.logger.logo(tb.format_exc())
                return 1.0
        elif self.dist_type == norm:
            mean, var = norm.fit(np.abs(np.subtract(actual, expected)))
            if var <= 0.0:
                var = 1e-20
            prob = norm.cdf(tol_thresh, loc=mean, scale=var)
        else:
            ecdf = ECDF(np.abs(np.subtract(actual, expected)) / np.abs(expected))
            prob = ecdf([tol_thresh])[0]
        return 1 - np.min(prob)

    # absolute tolerance
    def check_assert_tolerance(self, actual: list, expected: list, tol_thresh):
        if isinstance(tol_thresh, dict):
            if 'rtol' in tol_thresh:
                tol_thresh = tol_thresh['rtol']
            elif 'decimal' in tol_thresh:
                tol_thresh = tol_thresh['decimal']
            elif 'significant' in tol_thresh:
                tol_thresh = tol_thresh['significant']
            elif 'prec' in tol_thresh:
                tol_thresh = tol_thresh['prec']

        if self.enable_per_dim_comparison and Util.getdims(actual) >= 2:
            # for each index of actual, figure out the probability
            probs = []
            for ind in range(len(actual[0])):
                p = self.check_assert_tolerance([x[ind] for x in actual if ind < len(x)],
                                                expected if Util.getdims(expected) == 1 else [t[ind] for t in expected
                                                                                              if ind < len(t)],
                                                tol_thresh)
                probs.append(p)
            return np.max(probs)  # max probability of failing

        if self.dist_type is None:
            try:
                name, dist = self.compute_fit_score(np.abs(np.subtract(actual, expected)))
                prob = dist.cdf(tol_thresh)
            except:
                import traceback as tb
                self.logger.logo(tb.format_exc())
                return 1.0

        elif self.dist_type == norm:
            mean, var = norm.fit(np.abs(np.subtract(actual, expected)))
            if var <= 0.0:
                var = 1e-20
            prob = norm.cdf(tol_thresh, loc=mean, scale=var)
        else:
            ecdf = ECDF(np.abs(np.subtract(actual, expected)))
            prob = ecdf([tol_thresh])[0]
        return 1 - np.min(prob)

import ast
import multiprocessing
import os
import re
import subprocess as sp

import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

from CompareDistribution import CompareDistribution
from src import Util
from src.Config import Config
from src.SamplesEvaluator import SamplesEvaluator
from src.Util import sample_to_str
from src.lib.AssertSpec import AssertSpec


class TestDriver:
    def __init__(self, assert_spec: AssertSpec, rundir,
                 libdir,
                 parallel=False,
                 logstring="log>>>",
                 condaenvname="py3.6_allennlp",
                 threads=None,
                 logger=None
                 ):
        self.assert_spec = assert_spec
        self.parallel = parallel
        self.config = Config()
        self.patch = None
        self.logdir = None
        self.logstring = logstring
        self.logger = logger
        self.condaenvname = condaenvname
        self.rundir = rundir
        self.libdir = libdir
        self.threads = threads
        if self.parallel:
            exceptions = open("src/exceptionlist.txt").readlines()
            for l in exceptions:
                if assert_spec.test.filename.endswith(l.split(",")[1]) and assert_spec.test.classname == l.split(",")[
                    2]:
                    self.parallel = False
                    print("Found in exception list")

                    # decide on the format

    def parse_output(self, out, err):
        try:
            # assuming at least one match
            matched = re.findall("{0}([0-9.eE+,\[\]\(\) -]+)".format(self.logstring), str(out))
            if len(matched) == 0:
                matched = re.findall("{0}([0-9.eE+,\[\]\(\) -]+)".format(self.logstring), str(err))
            values = [np.array(ast.literal_eval(x)) if ('[' in x or '(' in x) else float(x) for x in matched]
            assert len(values) >= 2
            if len(values) >= 2:
                actual = np.array([values[i] for i in range(0, len(values), 2)])
                expected = np.array([values[i] for i in range(1, len(values), 2)])
                values = [actual, expected]

            return values
        except:
            self.logger.logo('Not found ::: ')
            return [np.finfo(np.float32).min, np.finfo(np.float32).min]

    def fetch_seeds(self, out):
        try:
            # assuming at least one match
            torchseed = re.findall("torch seed: ([0-9.eE-]*)", str(out))[0]
            numpyseed = re.findall("numpy seed: ([0-9.eE-]*)", str(out))[0]
            return {'torch_seed': torchseed, 'numpy_seed': numpyseed}
        except:
            self.logger.logo('Not found')
            return {'torch_seed': 0, 'numpy_seed': 0}

    def run_test(self):
        args = [self.assert_spec.test.get_test_str(), "-W", "ignore", "--capture=no"]
        output, error, returncode = self.run_pytest(args)
        return output, error, returncode

    def run_pytest(self, args) -> (str, str):
        result = sp.run(['{4}/src/runtest.sh {0} {1} {2} {3}'.format(
            self.assert_spec.test.filename,
            self.assert_spec.test.classname,
            self.assert_spec.test.testname,
            self.condaenvname,
            self.libdir
        )],
            shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
        return result.stdout, result.stderr, result.returncode

    def run_test_wrapper(self, i):
        print("Iter %s..." % i)
        return self.run_test()

    def mp_handler(self, data):
        if self.threads is not None:
            self.logger.logo("Launching %d jobs, %d in parallel" % (len(data), self.threads))
            with multiprocessing.Pool(self.threads) as p:
                results = p.map(self.run_test_wrapper, data)
        else:
            # default thread behaviour
            self.logger.logo(
                "Launching %d jobs, %d in parallel" % (len(data), self.config.THREAD_COUNT if self.parallel else 1))
            with multiprocessing.Pool(self.config.THREAD_COUNT if self.parallel else 1) as p:
                results = p.map(self.run_test_wrapper, data)
        return results

    @staticmethod
    def _append_samples(file, samples):
        with open(file, 'a') as samplefile:
            for sample in samples:
                # writing actual and expected
                for i in range(0, len(sample), 2):
                    samplefile.write("%s::%s\n" % (sample_to_str(sample[i]), sample_to_str(sample[i + 1])))

    @staticmethod
    def _write_results(results, base_iters, resultdir):
        for i in range(base_iters, base_iters + len(results)):
            with open(os.path.join(resultdir, 'output_{0}'.format(i)), 'w+') as outfile:
                outfile.write(results[i - base_iters][0].decode("utf-8"))
                outfile.write(results[i - base_iters][1].decode("utf-8"))

    def _write_report(self, outputs, extracted_outputs, errors, codes, convergence_scores):
        with open(os.path.join(self.logdir, 'report.txt'), 'w+') as report_file:
            report_file.write("Iterations: %d\n" % len(outputs))
            report_file.write("Passed : %d\n" % (sum([int(x) == 0 for x in codes])))
            report_file.write("Failed : %d\n" % (sum([int(x) != 0 for x in codes])))
            report_file.write("Convergence scores: %s\n" % ' '.join([str(x) for x in convergence_scores]))
            report_file.write("")
            report_file.write(Util.samples_stat(extracted_outputs, self.assert_spec))

    # main loop
    # start with N test executions, collect samples, evaluate convergence, decide when to stop
    def run_test_loop(self):
        # create a directory for logs
        self.logdir = os.path.join(self.rundir, "assert_{0}".format(self.assert_spec.get_hash()))
        self.logger.logo("Logdir: %s" % self.logdir)
        if os.path.exists(os.path.join(self.logdir, "report.txt")):
            self.logger.logo("Assertion exists.. exiting... ")
            self.logdir = None  # so that instrumentor does not write the files here
            return
        os.makedirs(self.logdir)

        samplefile_path = os.path.join(self.logdir, 'samples.txt')
        with open(samplefile_path, 'w+') as samplefile:
            samplefile.write(self.assert_spec.print_spec())
        outputs = []
        extracted_outputs = []
        errors = []
        codes = []
        convergence_scores = []
        decisions = []
        sampling_iterations = 0
        next_batch_iterations = self.config.DEFAULT_ITERATIONS

        while sampling_iterations < self.config.MAX_ITERATIONS:
            # run the test N times
            results = self.mp_handler(list(range(next_batch_iterations)))
            self._write_results(results, sampling_iterations, self.logdir)
            # collect results
            newsamples = []
            for r in results:
                outputs.append(r[0])
                parsed_outputs = self.parse_output(r[0], r[1])

                newsamples.append(parsed_outputs)
                errors.append(r[1])
                if int(r[2]) != 0:
                    if len(re.findall("1 passed", str(r[0].decode("utf-8")))) == 0:
                        # guarding against spurios failures, like thread failures
                        codes.append(r[2])
                    else:
                        codes.append(0)
                else:
                    codes.append(r[2])

            print("Passed tests : %d" % (sum([int(x) == 0 for x in codes])))
            print("Failed tests : %d" % (sum([int(x) != 0 for x in codes])))

            # update samples in file

            self._append_samples(samplefile_path, newsamples)

            # update samples in list
            extracted_outputs = extracted_outputs + newsamples


            # determine convergence
            samplesEvaluator = SamplesEvaluator([x[0] for x in extracted_outputs], self.config)
            converged, geweke_score = samplesEvaluator.check_covergence()
            convergence_scores.append(geweke_score)
            print("Converged: %s" % converged)
            if converged:
                break

            # update iterations
            sampling_iterations += next_batch_iterations
            next_batch_iterations = self.config.SUBSEQUENT_ITERATIONS
            print("Continuing to next batch...")

        compareDist = CompareDistribution(None,
                                          [x[0] for x in extracted_outputs],
                                          [x[1] for x in extracted_outputs],
                                          dist_type=ECDF,
                                          assert_type=self.assert_spec.assert_type,
                                          assert_str=self.assert_spec.assert_string,
                                          logger=self.logger)
        compareDist.evaluate()

        self._write_report(outputs, extracted_outputs, errors, codes, convergence_scores)

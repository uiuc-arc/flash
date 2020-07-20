#!/usr/bin/env python3
import argparse

import libraries
from Logger import Logger
from assertscraper import AssertScraper
from src.TestDriver import TestDriver
from src.TestInstrumentor import TestInstrumentor

parser = argparse.ArgumentParser(description='Flash arguments')
parser.add_argument('--replay', dest='replay', nargs='+',
                    default=list())  # restart using existing folder e.g. --replay tensor2tensor:<location>
parser.add_argument('-ct', dest='custom_threads', action='store_true')

args = parser.parse_args()
print(args)
replay_info = dict()
if 'replay' in args:
    for r in args.replay:
        replay_info[r.split(":")[0]] = r.split(":")[1]

# input : library
# input : assert patterns maybe
from src.Util import create_new_dir

libs = libraries.LIBRARIES

for library in libs:
    if not library.enabled:
        continue
    print("Testing library [%s]" % library.name)
    threads = None
    if args.custom_threads and 'threads' in library.__dict__:
        threads = library.threads
        print("Setting threads for {0} at {1}".format(library.name, library.threads))

    # mine assertions
    assert_scraper = AssertScraper(library.path, library.name)
    assert_scraper.parse_test_files()
    assert_scraper.filter_asserts()

    assertion_specs = assert_scraper.asserts

    if len(assertion_specs) == 0:
        print("No assertions found")
        continue

    # use the existing dir
    if library.name in replay_info:
        rundir = replay_info[library.name]
    else:
        rundir = create_new_dir("{0}/logs".format(libraries.PROJECT_DIR), "run_", '_' + library.name)
    logger = Logger(basedir=rundir)
    print("Rundir: %s" % rundir)
    for i, spec in enumerate(assertion_specs):
        # instrument the test
        logger.logo("Spec %d " % (i + 1))
        logger.logo(spec.print_spec())

        try:
            instrumentor = TestInstrumentor(spec, logstring='log>>>', deps=library.deps)
            instrumentor.instrument()
            instrumentor.write_file()

            # samples values from test
            testdriver = TestDriver(spec,
                                    parallel=library.parallel,
                                    condaenvname=library.conda_env,
                                    rundir=rundir,
                                    libdir=libraries.PROJECT_DIR,
                                    threads=threads,
                                    logger=logger)
            testdriver.run_test_loop()

            # save a copy of file and restore original file
            instrumentor.restore_file(testdriver.logdir)

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.logo(e)

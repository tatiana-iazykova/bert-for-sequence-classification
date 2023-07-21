import multiprocessing
import os
import sys
import unittest
from pathlib import Path

import coverage


def test_suite(tests_root_dir: Path):
    loader = unittest.TestLoader()

    suite = unittest.TestSuite()

    suite.addTest(loader.discover(str(tests_root_dir)))

    return suite


if __name__ == '__main__':
    cov = coverage.Coverage(omit='*/venv*/,example/,*/.github/')
    cov.start()
    if sys.platform.startswith('darwin'):
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            print("Tried to set multiprocessing context to 'spawn', but it appears that context is already set")
            raise

    cwd = Path.cwd()
    tests_root_dir = cwd / "bert_clf/tests"
    os.chdir(str(tests_root_dir))

    runner = unittest.TextTestRunner()
    runner.run(test_suite(tests_root_dir))

    cov.report()

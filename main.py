#!/usr/bin/env python3

from heti_stack import *

for test_group in [single_tests, multi_tests]:
	anneal_tests = [t for t in run_tests(test_group) if t.anneal]
	write_results(anneal_tests, "unhappiness_{0}".format(test_group.__name__),
		"results_{0}.org".format(test_group.__name__))
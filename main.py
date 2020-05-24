#!/usr/bin/env python3

from heti_stack import *

tests_to_run = [
	# ("single_tests", single_tests),
	("mixed_tests", mixed_tests)
]

for name, test_group in tests_to_run:
	anneal_tests = [t for t in run_tests(test_group, name, run_non_anneals=False,
				 	filters=[wanted_top_4_hospital, wanted_top_6_hospital]) if t.anneal]
	write_results(anneal_tests, "unhappiness_{0}".format(name),
		"results_{0}.org".format(name), mixed=name=="mixed_tests", filters=got_top_4_hospital)
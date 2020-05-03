#!/usr/bin/env python3

from heti_stack import *

for name, test_group in [("mixed_tests", mixed_tests)]:
	anneal_tests = [t for t in run_tests(test_group, name, filter_f=got_top_4_hospital) if t.anneal]
	write_results(anneal_tests, "unhappiness_{0}".format(name),
		"results_{0}.org".format(name), mixed=name=="mixed_tests", filter_f=got_top_4_hospital)
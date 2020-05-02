#!/usr/bin/env python3

from stack_aux import *
from org_write import *
from stack_sim import unhappy_df

tests = []
for test_name, test_f in single_test_function_dict.items():
	for i in [True, False]:
		print(test_name + (" (anneal)" if i else ""))
		current_test = Test(test_name, test_f, i)
		unhappy_df = unhappy_df.append({"alloc_mode": test_name, "anneal": i,
			"global_unhappiness": current_test.sim.current_unhappiness()}, ignore_index=True)
		if i:
			current_test.convergence()
		current_test.plot()
		current_test.export()
		tests.append(current_test)

unhappy_df.to_csv("tables/unhappiness_single.csv")

anneal_tests = [t for t in tests if t.anneal]
write_results(anneal_tests, "unhappiness_single", "results.org")
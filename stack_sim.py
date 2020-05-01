#!/usr/bin/env python3

from stack_aux import *

unhappy_df = pd.DataFrame(columns=["alloc_mode", "anneal", "global_unhappiness"])

# Code needs to be cleaned up for redundancies

test_function_dict = {
	"All random": shuffle,
	"Weighted random": weighted_shuffle,
	"All stack": default_stack,
	"All stack with random first": push_random_to_top,
	"All stack with weighted random first": push_wt_random_to_top,
	"All stack with random first and 12th": lambda l: push_random_to_top_and_n(l, 11),
	"All stack with random first and 14th": lambda l: push_random_to_top_and_n(l, 13),
	"All stack with weighted random first and 12th": lambda l: push_wt_random_to_top_and_n(l, 11),
	"All stack with weighted random first and 14th": lambda l: push_wt_random_to_top_and_n(l, 13),
}

tests = []
for test_name, test_f in test_function_dict.items():
	for i in [True, False]:
		print(test_name + " (anneal)" if i else "")
		current_test = Test(test_name, test_f, i)
		unhappy_df = unhappy_df.append({"alloc_mode": test_name, "anneal": i,
			"global_unhappiness": current_test.sim.current_unhappiness()}, ignore_index=True)
		if i:
			current_test.convergence()
		current_test.plot()
		current_test.export()
		tests.append(current_test)

unhappy_df.to_csv("tables/unhappiness.csv")
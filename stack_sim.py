#!/usr/bin/env python3

from stack_aux import *

unhappy_df = pd.DataFrame(columns=["alloc_mode", "anneal", "global_unhappiness"])

test_function_dict = {
	"All stack with weighted random first and 14th": lambda l: push_wt_random_to_top_and_n(l, 13),
	"All stack with weighted random first and 12th": lambda l: push_wt_random_to_top_and_n(l, 11),
	"All stack with weighted random first and 2nd": lambda l: push_wt_random_to_top_and_n(l, 1),
	"All stack with random first and 12th": lambda l: push_random_to_top_and_n(l, 11),
	"All stack with random first and 14th": lambda l: push_random_to_top_and_n(l, 13),
	"All stack with random first and 2nd": lambda l: push_random_to_top_and_n(l, 1),
	"All random": shuffle,
	"Weighted random": weighted_shuffle,
	"All stack": default_stack,
	"All stack with random first": push_random_to_top,
	"All stack with weighted random first": push_wt_random_to_top,
}

tests = []
for test_name, test_f in test_function_dict.items():
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

unhappy_df.to_csv("tables/unhappiness.csv")

anneal_tests = [t for t in tests if t.anneal]
with open("results.org", "w") as outfile:
	print("* Results", file=outfile)
	weighted_tests = [k for k in anneal_tests if "weighted" in k.name.lower()]
	unweighted_tests = [k for k in anneal_tests if "weighted" not in k.name.lower()]
	test_segregations = {
		"weighted": weighted_tests,
		"unweighted": unweighted_tests
	}
	for k, d in test_segregations.items():
		print("** AI algorithm + {0} random selection".format(k), file=outfile)
		for test in d:
			print("*** {0}".format(test.name), file=outfile)
			print("[[./images/{0}_satisfied.png]]".format(test.underscore_name), file=outfile)
			print(file=outfile)
			print(open("./tables/{0}.csv".format(test.underscore_name)).read(),file=outfile)
			for i in range(3):
				if i == 0:
					print("**** Total", file=outfile)
					print("[[./images/{0}_satisfied_total.png]]".format(test.underscore_name), file=outfile)
				else:
					print("**** Category {0}".format(i), file=outfile)
					print("[[./images/{0}_satisfied_cat{1}.png]]".format(test.underscore_name, i), file=outfile)
	print("** AI algorithm convergence", file=outfile)
	for test in anneal_tests:
		print("*** {0}".format(test.name), file=outfile)
		print("[[./images/conv_{0}.png]]".format(test.underscore_name), file=outfile)
	print("** Global unhappiness when compared to categorical matching", file=outfile)
	print(open("./tables/unhappiness.csv").read(),file=outfile)
#!/usr/bin/env python3

"""
Functionality relevant to running tests
"""

from .sim import *
from .base import underscorify

# Test object

class Test(object):
	"""Holds information about a stacking strategy test"""
	def __init__(self, name: str, function, anneal: bool):
		self.name = name
		self.underscore_name = underscorify(name) + ("_anneal" if anneal else "")
		self.function = function
		self.anneal = anneal
		self.sim = AnnealSimulation(function) if anneal else Simulation(function)
	def convergence(self):
		if self.anneal:
			self.sim.unhappiness_records.plot.line()
			plt.xlabel("Number of iterations")
			plt.ylabel("Global unhappiness")
			plt.title('\n'.join(wrap("Convergence - {0}".format(self.name), 60)))
			plt.tight_layout()
			# plt.savefig("images/"+"conv_"+self.underscore_name+".png", dpi=300)
			plt.show()
			plt.clf()
			plt.cla()
			plt.close('all')
		else:
			raise Exception
	def plot(self, filter_f=None):
		title_insert = " (filter by {0})".format(filter_function_names[filter_f.__name__].lower()) if filter_f else ""
		filename_insert = "_filter_{0}".format(filter_f.__name__) if filter_f else ""
		self.sim.plot_every(filter_f=filter_f, prepend=self.name+title_insert+": ", filename_pre=self.underscore_name+filename_insert)
		self.sim.plot_all(filter_f=filter_f, prepend=self.name+title_insert+": ", filename_pre=self.underscore_name+filename_insert)
	def export(self, filter_f=None):
		self.sim.export_results(self.underscore_name, filter_f=filter_f)
	def sigtest(self, groups=None, filter_f=None):
		if groups:
			print(self.sim.compare_two_subgroups(groups, filter_f))
			print(self.sim.compare_two_firsts(groups, filter_f))
		print(self.sim.compare_all_subgroups(filter_f))

# Test dictionaries
# TODO: consider changing the test dictionaries to lists of Test objects
# TODO: add filters for likelihood of getting into top 4 and top 6 hospitals

single_tests = {
	# "All random": shuffle,
	"Weighted random": weighted_shuffle,
	"All stack": default_stack,
	"Mixed stacks": mixed_stacks,
	# "All same stack with random first": lambda l: push_random_to_top(stack),
	"All same stack with weighted random first": lambda l: push_wt_random_to_top(stack),
	# "All same stack with random first and 12th": lambda l: push_random_to_top_and_n(stack, 11),
	"All same stack with weighted random first and 12th": lambda l: push_wt_random_to_top_and_n(stack, 11),
	# "All same stack with random first and 14th": lambda l: push_random_to_top_and_n(stack, 13),
	"All same stack with weighted random first and 14th": lambda l: push_wt_random_to_top_and_n(stack, 13),
	# "All same stack with random first and 2nd": lambda l: push_random_to_top_and_n(stack, 1),
	"All same stack with weighted random first and 2nd": lambda l: push_wt_random_to_top_and_n(stack, 1),
	# "Mixed stacks with random first": lambda l: push_random_to_top(mixed_stacks(l)),
	"Mixed stacks with weighted random first": lambda l: push_wt_random_to_top(mixed_stacks(l)),
	# "Mixed stacks with random first and 12th": lambda l: push_random_to_top_and_n(mixed_stacks(l), 11),
	"Mixed stacks with weighted random first and 12th": lambda l: push_wt_random_to_top_and_n(mixed_stacks(l), 11),
	# "Mixed stacks with random first and 14th": lambda l: push_random_to_top_and_n(mixed_stacks(l), 13),
	"Mixed stacks with weighted random first and 14th": lambda l: push_wt_random_to_top_and_n(mixed_stacks(l), 13),
	# "Mixed stacks with random first and 2nd": lambda l: push_random_to_top_and_n(mixed_stacks(l), 1),
	"Mixed stacks with weighted random first and 2nd": lambda l: push_wt_random_to_top_and_n(mixed_stacks(l), 1),
}

mixed_tests = {
	"Mixed strategies: 80% stack with weighted random first, 20% weighted random": [
		(stack_wt_random_top, 0.8), (weighted_shuffle, 0.2)
		],
	# "Mixed strategies: 80% stack with unweighted random, 20% unweighted random": [
	# 	(stack_random_top, 0.8), (shuffle, 0.2)
	# 	],
	"Mixed strategies: 60% stack with weighted random, 40% weighted random": [
		(stack_wt_random_top, 0.6), (weighted_shuffle, 0.4)
		],
	# "Mixed strategies: 60% stack with unweighted random, 40% unweighted random": [
	# 	(stack_random_top, 0.6), (shuffle, 0.4)
	# 	],
	"Mixed strategies: 10% random with weighted random top, 30% mixed stacks, 60% default stack": [
		(random_with_wt_random_top, 0.10), (mixed_stacks, 0.3), (default_stack, 0.6)
	],
	"Mixed strategies: 15% random with weighted random top, 30% mixed stacks with weighted random top, 55% default stack with weighted random top": [
		(random_with_wt_random_top, 0.15), (mixed_stack_wt_random_top, 0.3), (stack_wt_random_top, 0.55)
	],
	"Mixed strategies: 20% random with weighted random top, 25% mixed stacks with weighted random top, 55% default stack with weighted random top": [
		(random_with_wt_random_top, 0.20), (mixed_stack_wt_random_top, 0.25), (stack_wt_random_top, 0.55)
	],
	"Mixed strategies: 20% random with weighted random top, 50% mixed stacks with weighted random top, 30% default stack with weighted random top": [
		(random_with_wt_random_top, 0.20), (mixed_stack_wt_random_top, 0.50), (stack_wt_random_top, 0.30)
	],
	"Mixed strategies: 50% random with weighted random top, 25% mixed stacks with weighted random top, 25% default stack with weighted random top": [
		(random_with_wt_random_top, 0.50), (mixed_stack_wt_random_top, 0.25), (stack_wt_random_top, 0.25)
	],
}

def run_tests(function_dict, name, run_non_anneals=True, mixed=False, filters=[]):
	# print("Filter function:", filter_f)
	unhappy_df = pd.DataFrame(columns=["alloc_mode", "anneal", "global_unhappiness"])
	tests = []
	anneal_switches = [True] + ([False] if run_non_anneals else [])
	for test_name, test_f in function_dict.items():
		for i in anneal_switches:
			print(test_name + (" (anneal)" if i else ""))
			current_test = Test(test_name, test_f, i)
			unhappy_df = unhappy_df.append({"alloc_mode": test_name, "anneal": i,
				"global_unhappiness": current_test.sim.current_unhappiness()}, ignore_index=True)
			if i:
				current_test.convergence()
			current_test.plot()
			current_test.export()
			current_test.sigtest([["random_with_wt_random_top"], ["mixed_stack_wt_random_top",
				"stack_wt_random_top"]])
			for filter_f in filters:
				print("Filtering by {0}".format(filter_f.__name__))
				current_test.plot(filter_f)
				current_test.export(filter_f)
				current_test.sigtest([["random_with_wt_random_top"], ["mixed_stack_wt_random_top",
				"stack_wt_random_top"]], filter_f)
			tests.append(current_test)
	unhappy_df.to_csv("tables/unhappiness_{0}.csv".format(name))
	return tests

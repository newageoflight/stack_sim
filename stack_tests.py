#!/usr/bin/env python3

from stack_sim import *
from org_write import *

# Test object

class Test(object):
	"""Holds information about a stacking strategy test"""
	def __init__(self, name: str, function, anneal: bool):
		self.name = name
		self.underscore_name = Test.underscorify(name) + ("_anneal" if anneal else "")
		self.function = function
		self.anneal = anneal
		self.sim = AnnealSimulation(function) if anneal else Simulation(function)
	@staticmethod
	def underscorify(name):
		new_name = re.sub(r"[\s]", "_", name)
		new_name = sanitise_filename(new_name.lower())
		return new_name
	def convergence(self):
		if self.anneal:
			self.sim.unhappiness_records.plot.line()
			plt.xlabel("Number of iterations")
			plt.ylabel("Global unhappiness")
			plt.title("Convergence - {0}".format(self.name))
			plt.tight_layout()
			plt.savefig("images/"+"conv_"+self.underscore_name+".png", dpi=300)
			# plt.show()
			plt.clf()
			plt.cla()
			plt.close('all')
		else:
			raise Exception
	def plot(self):
		self.sim.plot_every(prepend=self.name+": ", filename_pre=self.underscore_name)
		self.sim.plot_all(prepend=self.name+": ", filename_pre=self.underscore_name)
	def export(self):
		self.sim.export_results(self.underscore_name)

# Test dictionaries

single_tests = {
	"All random": shuffle,
	"Weighted random": weighted_shuffle,
	"All stack": default_stack,
	"Mixed stacks": mixed_stacks,
	"All same stack with random first": lambda l: push_random_to_top(stack),
	"All same stack with weighted random first": lambda l: push_wt_random_to_top(stack),
	"All same stack with random first and 12th": lambda l: push_random_to_top_and_n(stack, 11),
	"All same stack with weighted random first and 12th": lambda l: push_wt_random_to_top_and_n(stack, 11),
	"All same stack with random first and 14th": lambda l: push_random_to_top_and_n(stack, 13),
	"All same stack with weighted random first and 14th": lambda l: push_wt_random_to_top_and_n(stack, 13),
	"All same stack with random first and 2nd": lambda l: push_random_to_top_and_n(stack, 1),
	"All same stack with weighted random first and 2nd": lambda l: push_wt_random_to_top_and_n(stack, 1),
	"Mixed stacks with random first": lambda l: push_random_to_top(mixed_stacks(l)),
	"Mixed stacks with weighted random first": lambda l: push_wt_random_to_top(mixed_stacks(l)),
	"Mixed stacks with random first and 12th": lambda l: push_random_to_top_and_n(mixed_stacks(l), 11),
	"Mixed stacks with weighted random first and 12th": lambda l: push_wt_random_to_top_and_n(mixed_stacks(l), 11),
	"Mixed stacks with random first and 14th": lambda l: push_random_to_top_and_n(mixed_stacks(l), 13),
	"Mixed stacks with weighted random first and 14th": lambda l: push_wt_random_to_top_and_n(mixed_stacks(l), 13),
	"Mixed stacks with random first and 2nd": lambda l: push_random_to_top_and_n(mixed_stacks(l), 1),
	"Mixed stacks with weighted random first and 2nd": lambda l: push_wt_random_to_top_and_n(mixed_stacks(l), 1),
}

mixed_tests = {
	"Mixed strategies: 80% stack, 20% weighted random": mixed_strategy_stack_and_wt_random(),
	"Mixed strategies: 80% stack, 20% unweighted random": mixed_strategy_stack_and_random()
}

def run_tests(function_dict, run_non_anneals=True, mixed=False):
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
			tests.append(current_test)
	unhappy_df.to_csv("tables/unhappiness_{0}.csv".format(function_dict.__name__))
	return tests
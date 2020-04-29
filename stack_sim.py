#!/usr/bin/env python3

from stack_aux import *

def main():
	print("""
If all applicants select completely randomly:
------------------
	""")
	applicants_all_random = Simulation(shuffle)
	applicants_all_random.plot_all(prepend="All random: ")
	applicants_all_random.plot_every(prepend="All random: ")
	applicants_all_random.export_results("all_random")

	print("""
If all applicants stack:
------------------
	""")
	applicants_all_stacked = Simulation(default_stack)
	applicants_all_stacked.plot_all(prepend="All stack: ")
	applicants_all_stacked.plot_every(prepend="All stack: ")
	applicants_all_stacked.export_results("all_stack")

	print("""
If all applicants stack but move a random hospital to first:
------------------
	""")
	applicants_stacked_with_random_first = Simulation(push_random_to_top)
	applicants_stacked_with_random_first.plot_all(prepend="All stack but put random at top: ")
	applicants_stacked_with_random_first.plot_every(prepend="All stack but put random at top: ")
	applicants_stacked_with_random_first.export_results("all_stack_top_random")

	print("""
If all applicants stack but move a random hospital to first and 14th:
------------------
	""")
	applicants_stacked_with_random_first = Simulation(push_random_to_top_and_14)
	applicants_stacked_with_random_first.plot_all(prepend="All stack but put random at top and 14: ")
	applicants_stacked_with_random_first.plot_every(prepend="All stack but put random at top and 14: ")
	applicants_stacked_with_random_first.export_results("all_stack_top_random_and_14")

if __name__ == '__main__':
	main()
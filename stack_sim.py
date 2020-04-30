#!/usr/bin/env python3

from stack_aux import *

unhappy_df = pd.DataFrame(columns=["alloc_mode", "anneal", "global_unhappiness"])

# Code needs to be cleaned up for redundancies

print("All random (anneal)")
applicants_all_random_anneal = AnnealSimulation(shuffle)
unhappy_df = unhappy_df.append({"alloc_mode": "All random", "anneal": True,
	"global_unhappiness": applicants_all_random_anneal.current_unhappiness()}, ignore_index=True)
applicants_all_random_anneal.unhappiness_records.plot.line()
plt.title("Convergence - all random (anneal)")
plt.tight_layout()
plt.savefig("images/"+"Convergence - all random (anneal)"+".png", dpi=300)
# plt.show()
plt.clf()
plt.cla()
plt.close('all')
applicants_all_random_anneal.plot_all(prepend="All random (anneal): ")
applicants_all_random_anneal.plot_every(prepend="All random (anneal): ")
applicants_all_random_anneal.export_results("all_random_anneal")

print("All random")
applicants_all_random = Simulation(shuffle)
unhappy_df = unhappy_df.append({"alloc_mode": "All random", "anneal": False,
	"global_unhappiness": applicants_all_random.current_unhappiness()}, ignore_index=True)
applicants_all_random.plot_all(prepend="All random: ")
applicants_all_random.plot_every(prepend="All random: ")
applicants_all_random.export_results("all_random")

print("Weighted random (anneal)")
applicants_weighted_random_anneal = AnnealSimulation(weighted_shuffle)
unhappy_df = unhappy_df.append({"alloc_mode": "Weighted random", "anneal": True,
	"global_unhappiness": applicants_weighted_random_anneal.current_unhappiness()}, ignore_index=True)
applicants_weighted_random_anneal.unhappiness_records.plot.line()
plt.title("Convergence - weighted random (anneal)")
plt.tight_layout()
plt.savefig("images/"+"Convergence - weighted random (anneal)"+".png", dpi=300)
# plt.show()
plt.clf()
plt.cla()
plt.close('all')
applicants_weighted_random_anneal.plot_all(prepend="Weighted random (anneal): ")
applicants_weighted_random_anneal.plot_every(prepend="Weighted random (anneal): ")
applicants_weighted_random_anneal.export_results("weighted_random_anneal")

print("Weighted random")
applicants_weighted_random = Simulation(weighted_shuffle)
unhappy_df = unhappy_df.append({"alloc_mode": "Weighted random", "anneal": False,
	"global_unhappiness": applicants_weighted_random.current_unhappiness()}, ignore_index=True)
applicants_weighted_random.plot_all(prepend="Weighted random: ")
applicants_weighted_random.plot_every(prepend="Weighted random: ")
applicants_weighted_random.export_results("weighted_random")

print("All stack (anneal)")
applicants_all_stacked_anneal = AnnealSimulation(default_stack)
unhappy_df = unhappy_df.append({"alloc_mode": "All stack", "anneal": True,
	"global_unhappiness": applicants_all_stacked_anneal.current_unhappiness()}, ignore_index=True)
applicants_all_stacked_anneal.unhappiness_records.plot.line()
plt.title("Convergence - all stack (anneal)")
plt.tight_layout()
plt.savefig("images/"+"Convergence - all stack (anneal)"+".png", dpi=300)
# plt.show()
plt.clf()
plt.cla()
plt.close('all')
applicants_all_stacked_anneal.plot_all(prepend="All stack (anneal): ")
applicants_all_stacked_anneal.plot_every(prepend="All stack (anneal): ")
applicants_all_stacked_anneal.export_results("all_stack_anneal")

print("All stack")
applicants_all_stacked = Simulation(default_stack)
unhappy_df = unhappy_df.append({"alloc_mode": "All stack", "anneal": False,
	"global_unhappiness": applicants_all_stacked.current_unhappiness()}, ignore_index=True)
applicants_all_stacked.plot_all(prepend="All stack: ")
applicants_all_stacked.plot_every(prepend="All stack: ")
applicants_all_stacked.export_results("all_stack")

print("All stack with random first (anneal)")
applicants_stacked_with_random_first_anneal = AnnealSimulation(push_random_to_top)
unhappy_df = unhappy_df.append({"alloc_mode": "All stack with random first", "anneal": True,
	"global_unhappiness": applicants_stacked_with_random_first_anneal.current_unhappiness()}, ignore_index=True)
applicants_stacked_with_random_first_anneal.unhappiness_records.plot.line()
plt.title("Convergence - all stack, random first (anneal)")
plt.tight_layout()
plt.savefig("images/"+"Convergence - all stack random first (anneal)"+".png", dpi=300)
# plt.show()
plt.clf()
plt.cla()
plt.close('all')
applicants_stacked_with_random_first_anneal.plot_all(prepend="All stack but put random at top (anneal): ")
applicants_stacked_with_random_first_anneal.plot_every(prepend="All stack but put random at top (anneal): ")
applicants_stacked_with_random_first_anneal.export_results("all_stack_top_random_anneal")

print("All stack with random first")
applicants_stacked_with_random_first = Simulation(push_random_to_top)
unhappy_df = unhappy_df.append({"alloc_mode": "All stack with random first", "anneal": False,
	"global_unhappiness": applicants_stacked_with_random_first.current_unhappiness()}, ignore_index=True)
applicants_stacked_with_random_first.plot_all(prepend="All stack but put random at top: ")
applicants_stacked_with_random_first.plot_every(prepend="All stack but put random at top: ")
applicants_stacked_with_random_first.export_results("all_stack_top_random")

print("All stack with weighted random first (anneal)")
applicants_stacked_with_wt_random_first_anneal = AnnealSimulation(push_wt_random_to_top)
unhappy_df = unhappy_df.append({"alloc_mode": "All stack with weighted random first", "anneal": True,
	"global_unhappiness": applicants_stacked_with_wt_random_first_anneal.current_unhappiness()}, ignore_index=True)
applicants_stacked_with_wt_random_first_anneal.unhappiness_records.plot.line()
plt.title("Convergence - all stack, weighted random first (anneal)")
plt.tight_layout()
plt.savefig("images/"+"Convergence - all stack weighted random first (anneal)"+".png", dpi=300)
# plt.show()
plt.clf()
plt.cla()
plt.close('all')
applicants_stacked_with_wt_random_first_anneal.plot_all(prepend="All stack but put weighted random at top (anneal): ")
applicants_stacked_with_wt_random_first_anneal.plot_every(prepend="All stack but put weighted random at top (anneal): ")
applicants_stacked_with_wt_random_first_anneal.export_results("all_stack_top_wt_random_anneal")

print("All stack with weighted random first")
applicants_stacked_with_wt_random_first = Simulation(push_wt_random_to_top)
unhappy_df = unhappy_df.append({"alloc_mode": "All stack with weighted random first", "anneal": False,
	"global_unhappiness": applicants_stacked_with_wt_random_first.current_unhappiness()}, ignore_index=True)
applicants_stacked_with_wt_random_first.plot_all(prepend="All stack but put weighted random at top: ")
applicants_stacked_with_wt_random_first.plot_every(prepend="All stack but put weighted random at top: ")
applicants_stacked_with_wt_random_first.export_results("all_stack_top_wt_random")

unhappy_df.to_csv("tables/unhappines.csv")
# print("""
# If all applicants stack but move a random hospital to first and 14th (anneal):
# ------------------
# """)
# applicants_stacked_with_random_first_anneal = AnnealSimulation(push_random_to_top_and_14)
# print(applicants_all_random_anneal.current_unhappiness())
# applicants_stacked_with_random_first_anneal.plot_all(prepend="All stack but put random at top and 14 (anneal): ")
# applicants_stacked_with_random_first_anneal.plot_every(prepend="All stack but put random at top and 14 (anneal): ")
# applicants_stacked_with_random_first_anneal.export_results("all_stack_top_random_and_14_anneal")

# print("""
# If all applicants stack but move a random hospital to first and 14th:
# ------------------
# """)
# applicants_stacked_with_random_first = Simulation(push_random_to_top_and_14)
# print(applicants_all_random.current_unhappiness())
# applicants_stacked_with_random_first.plot_all(prepend="All stack but put random at top and 14: ")
# applicants_stacked_with_random_first.plot_every(prepend="All stack but put random at top and 14: ")
# applicants_stacked_with_random_first.export_results("all_stack_top_random_and_14")
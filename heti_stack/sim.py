#!/usr/bin/env python3

"""
ALlocation process simulation objects
"""

from copy import deepcopy
from functools import reduce
from matplotlib.ticker import PercentFormatter
from math import exp
from scipy import stats
from textwrap import wrap

from .base import *

import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd

plt.style.use("ggplot")
plt.rcParams["font.family"] = "Avenir Next"

# Simulation objects
# This is all because of the need for stratification by stacking strategy, represents a major overhaul

class Simulation(object):
	"""Runs a simulation of the allocation process
	starting_strategy can either be a single function with a single argument
	or a list of tuples: (function, weight)"""
	def __init__(self, starting_strategy, dra_prefill=False, rounds=1):
		self.category_counts = category_counts
		self.hospitals = hospitals.copy()
		for hospital in self.hospitals:
			hospital.empty()
		if callable(starting_strategy) or isinstance(starting_strategy, StrategyObj):
			self.strategies_used = [starting_strategy.__name__]
			self.applicants = [Applicant(starting_strategy, cat) for cat in range(len(category_counts)) for i in range(category_counts[cat])]
		elif type(starting_strategy) == list:
			functions, weights = list(zip(*starting_strategy))
			self.strategies_used = [f.__name__ for f in functions]
			self.applicants = [Applicant(choice(functions, 1, p=weights)[0], cat) for cat in range(len(category_counts)) for i in range(category_counts[cat])]
		else:
			raise TypeError
		self.allocation_rounds = rounds
		self.stratified_results = pd.DataFrame()
		self.stratified_filtered_results = {}
		self.filtered_results = {}
		self.results = pd.DataFrame()
		if dra_prefill:
			self._dra_prefill()
		self._runsim()
	def _dra_prefill(self):
		"""Prefill DRA-eligible hospitals with candidates who have preferenced them first prior to allocation round"""
		category_ones = [a for a in self.applicants if a.category == 0]
		for hospital in filter(lambda h: h.is_dra, self.hospitals):
			dra_candidates = [a for a in category_ones if a.preferences[0] == hospital]
			hospital.fill(dra_candidates, dra_prefill=True)
		# DRA spots are only given to category 2-4 after the optimised allocation process:
		# https://www.heti.nsw.gov.au/__data/assets/pdf_file/0011/424667/Direct-Regional-Allocation-Procedure.PDF
	def _runsim(self):
		self._dra_prefill()
		for category in range(len(self.category_counts)):
			for hospital in self.hospitals:
				for rank in range(len(self.hospitals)):
					preferenced_this = [a for a in self.unplaced(self.applicants) if a.preferences[rank] == hospital
										and a.category == category]
					hospital.fill(preferenced_this)
		self._make_results()
	@staticmethod
	def satisfied(apps, rank, category=None):
		return [a for a in apps if a.preference_number == rank and (a.category == category if category != None else True)]
	@staticmethod
	def placed(apps, category=None):
		return [a for a in apps if a.allocation!=None and (a.category == category if category != None else True)]
	@staticmethod
	def unplaced(apps, category=None):
		return [a for a in apps if a.allocation==None and (a.category == category if category != None else True)]
	def dra_only(self):
		return [a for a in self.applicants if a.is_dra]
	def non_dra_only(self):
		return [a for a in self.applicants if not a.is_dra]
	def stratify_applicants_by_strategy(self):
		return {s: [a for a in self.applicants if a.strategy == s] for s in self.strategies_used}
	def stratify_applicants_by_strategy_and_filter(self, test):
		return {s: [a for a in self.applicants if a.strategy == s and test(a)] for s in self.strategies_used}
	def filter_applicants(self, test):
		return [a for a in self.applicants if test(a)]
	def _make_results(self):
		df_dict = {}
		flags = ["total", "cat"]
		for flag in flags:
			for i in range(6 if flag=="cat" else 1):
				cat = i if flag == "cat" else None
				append_str = "" if flag=="total" else str(i+1)
				placed = len(self.placed(self.applicants, cat))
				not_placed = len(self.unplaced(self.applicants, cat))
				total = sum(self.category_counts) if flag == "total" else self.category_counts[cat]
				df_dict[flag+append_str] = [len(self.satisfied(self.applicants, rank, cat)) for rank in range(len(self.hospitals))] + [placed, not_placed, total]
		self.results = pd.DataFrame(df_dict, index=[ordinal(n) for n in range(1, len(self.hospitals)+1)]+["placed","not_placed", "total"])
		if len(self.strategies_used) > 1:
			self.stratify_results()
		return self.results
	def stratify_and_filter_results(self, test, keep_original_totals=False):
		# TODO: retain unfiltered data's totals
		# will allow for better calculation of strategy usefulness
		applicant_pools = self.stratify_applicants_by_strategy_and_filter(test)
		# print("Applicants stratified by strategy and filter:", applicant_pools)
		df_dict = {flag:[] for flag in ["total"]+["cat"+str(n+1) for n in range(6)]}
		flags = ["total", "cat"]
		indices = [ordinal(n) for n in range(1, len(self.hospitals)+1)]+["placed","not_placed", "total"]
		strategies = applicant_pools.keys()
		for strategy, pool in applicant_pools.items():
			for flag in flags:
				for i in range(6 if flag=="cat" else 1):
					subpool = pool if flag == "total" else [a for a in pool if a.category == i]
					append_str = "" if flag=="total" else str(i+1)
					placed = len(self.placed(subpool))
					not_placed = len(self.unplaced(subpool))
					total = len(subpool)
					df_dict[flag+append_str] += [len(self.satisfied(subpool, rank)) for rank in range(len(self.hospitals))] + [placed, not_placed, total]
		df = pd.DataFrame(df_dict, index=pd.MultiIndex.from_product([strategies, indices]))
		self.stratified_filtered_results[test.__name__] = df
		return df
	def stratify_results(self):
		applicant_pools = self.stratify_applicants_by_strategy()
		df_dict = {flag:[] for flag in ["total"]+["cat"+str(n+1) for n in range(6)]}
		flags = ["total", "cat"]
		indices = [ordinal(n) for n in range(1, len(self.hospitals)+1)]+["placed","not_placed", "total"]
		strategies = applicant_pools.keys()
		for strategy, pool in applicant_pools.items():
			for flag in flags:
				for i in range(6 if flag=="cat" else 1):
					subpool = pool if flag == "total" else [a for a in pool if a.category == i]
					append_str = "" if flag=="total" else str(i+1)
					placed = len(self.placed(subpool))
					not_placed = len(self.unplaced(subpool))
					total = len(subpool)
					df_dict[flag+append_str] += [len(self.satisfied(subpool, rank)) for rank in range(len(self.hospitals))] + [placed, not_placed, total]
		df = pd.DataFrame(df_dict, index=pd.MultiIndex.from_product([strategies, indices]))
		self.stratified_results = df
		return self.stratified_results
	def filter_results(self, test, keep_original_totals=False):
		# TODO: needs option to preserve the total being from the non-filtered dataset
		# doing so will be more useful for calculations of strategy usefulness
		filtered_applicants = self.filter_applicants(test)
		df_dict = {}
		flags = ["total", "cat"]
		for flag in flags:
			for i in range(6 if flag=="cat" else 1):
				cat = i if flag == "cat" else None
				append_str = "" if flag=="total" else str(i+1)
				placed = len(self.placed(filtered_applicants, cat))
				not_placed = len(self.unplaced(filtered_applicants, cat))
				if not keep_original_totals:
					total = len(filtered_applicants) if flag == "total" else len([a for a in filtered_applicants if a.category == i])
				else:
					total = len(self.applicants) if flag == "total" else len([a for a in self.applicants if a.category == i])
				df_dict[flag+append_str] = [len(self.satisfied(filtered_applicants, rank, cat)) for rank in range(len(self.hospitals))] + [placed, not_placed, total]
		filtered_results = pd.DataFrame(df_dict, index=[ordinal(n) for n in range(1, len(self.hospitals)+1)]+["placed","not_placed", "total"])
		self.filtered_results[test.__name__] = filtered_results
		return filtered_results
	@staticmethod
	def percentify_results(results):
		new_results = results.copy()
		new_results = 100. * new_results/new_results.iloc[-1]
		# print(results.to_string(), new_results.to_string())
		return new_results.iloc[:17]
	@staticmethod
	def percentify_stratified_results(results):
		# TODO: fix bug, keeps throwing them, also results look wrong
		to_ret = []
		new_results = results.copy()
		for idx, df in new_results.groupby(level=0):
			to_ret.append(Simulation.percentify_results(df))
		# print(pd.concat(to_ret))
		return pd.concat(to_ret)
	def export_results(self, name: str, filter_f=None):
		self.results.to_csv("tables/"+name+".csv")
		self.percentify_results(self.results).to_csv("tables/"+name+"_percentified.csv")
		if filter_f:
			self.filter_results(filter_f).to_csv("tables/"+name+"_filter_"+filter_f.__name__+".csv")
		if len(self.strategies_used) > 1:
			self.stratified_results.to_csv("tables/"+name+"_stratified.csv")
			self.percentify_stratified_results(self.stratified_results).to_csv("tables/"+name+"_stratified_percentified.csv")
			self._recut_results("cat1", False).to_csv("tables/"+name+"_stratified_recut.csv")
			if filter_f:
				stratify_and_filter = self.stratify_and_filter_results(filter_f)
				stratify_and_filter.to_csv("tables/"+name+"_stratified_filter_"+filter_f.__name__+".csv")
				self.percentify_stratified_results(stratify_and_filter).to_csv("tables/"+name+"_stratified_filter_"+filter_f.__name__+"_percentified.csv")
				self._recut_results("cat1", False, filter_f).to_csv("tables/"+name+"_stratified_filter_"+filter_f.__name__+"_recut.csv")
	def pprint(self):
		"""Legacy function: pretty-prints out the results"""
		for rank in range(len(self.hospitals)):
			satisfied = self.satisfied(self.applicants, rank)
			print("Total applicants who got their {ord} preference: {count} ({percent:.2%})".format(ord=ordinal(rank+1), count=len(satisfied), 
				percent=len(satisfied)/sum(self.category_counts)))
		placed = len(self.placed(self.applicants))
		not_placed = len(self.unplaced(self.applicants))
		total = sum(self.category_counts)
		print("Total applicants who received any placement: {count} ({percent:.2%})".format(count=placed, percent=placed/total))
		print("Total applicants who did not get any placement: {count} ({percent:.2%})".format(count=not_placed, percent=not_placed/total))
		print("Total applicants: {count}".format(count=sum(self.category_counts), percent=placed/total))
		for category in range(len(self.category_counts)):
			for rank in range(len(self.hospitals)):
				satisfied = self.satisfied(self.applicants, rank, category)
				print("Total Category {cat} applicants who got their {ord} preference: {count} ({percent:.2%})".format(ord=ordinal(rank+1), count=len(satisfied), 
					percent=len(satisfied)/category_counts[category], cat=category+1))
			cat_placed = len(self.placed(self.applicants, category))
			cat_not_placed = len(self.unplaced(self.applicants, category))
			cat_total = self.category_counts[category]
			print("Total Category {cat} applicants who received any placement: {count} ({percent:.2%})".format(cat=category+1, count=cat_placed, percent=cat_placed/cat_total))
			print("Total Category {cat} applicants who did not get any placement: {count} ({percent:.2%})".format(cat=category+1, count=cat_not_placed, percent=cat_not_placed/cat_total))
			print("Total Category {cat} applicants: {count}".format(cat=category+1, count=cat_total))
	def plot_one(self, header, percent=True, filter_f=None, prepend="", filename_pre=""):
		if len(self.strategies_used) > 1:
			self.plot_one_stratified(header, percent, filter_f, prepend, filename_pre)
		else:
			toplot = self.filter_results(filter_f) if filter_f != None else self.results
			toplot = self.percentify_results(toplot) if percent else toplot
			fig, ax = plt.subplots()
			ax.yaxis.set_major_formatter(PercentFormatter())
			title = prepend + "Satisfied applicants: {header}".format(header=header)
			filename = filename_pre + "_satisfied_{header}".format(header=header)
			toplot.plot.bar(y=header, rot=30)
			self._plot("Applicants who got their nth preference", "%" if percent else "count", title, filename)
	def plot_all(self, percent=True, filter_f=None, prepend="", filename_pre=""):
		if len(self.strategies_used) > 1:
			self.plot_all_stratified(percent, filter_f, prepend, filename_pre)
		else:
			toplot = self.filter_results(filter_f) if filter_f != None else self.results
			toplot = self.percentify_results(toplot) if percent else toplot
			toplot.plot.bar(rot=30)
			self._plot("Applicants who got their nth preference", "%" if percent else "count", prepend + "Satisfied applicants",
				filename_pre + "_satisfied")
	def plot_every(self, percent=True, filter_f=None, prepend="", filename_pre=""):
		if len(self.strategies_used) > 1:
			self.plot_every_stratified(percent, filter_f, prepend, filename_pre)
		else:
			for col in self.results:
				self.plot_one(col, percent, filter_f, prepend, filename_pre)
	def _recut_results(self, header, percent=False, filter_f=None):
		if filter_f != None:
			result_df = self.stratify_and_filter_results(filter_f)
		else:
			result_df = self.stratified_results
		result_df = self.percentify_stratified_results(result_df) if percent else result_df
		# print(result_df.to_string())
		toplot = result_df.xs(header, axis=1).unstack().T
		# print(toplot.to_string())
		toplot = toplot.reindex(result_df.index.get_level_values(1).unique())
		return toplot
	def plot_one_stratified(self, header, percent=True, filter_f=None, prepend="", filename_pre=""):
		"""Separate function for plotting single columns from stratified dataframes"""
		# it's now possible to plot a stratified single function using df.xs("a", axis=1).unstack().plot().bar()
		# none of this code is now functional so keep your hands off of it. may need to revert to a previous commit
		# TODO: fix bug, positions on chart are out of order
		toplot = self._recut_results(header, percent, filter_f)
		# print(toplot)
		fig, ax = plt.subplots()
		ax.yaxis.set_major_formatter(PercentFormatter())
		# print(toplot.to_string())
		toplot.plot.bar(rot=30)
		title = prepend + "Satisfied applicants: {header}".format(header=header)
		filename = filename_pre + "_satisfied_{header}".format(header=header)
		self._plot("Applicants who got their nth preference", "%" if percent else "count", title, filename)
		for col in toplot:
			toplot.plot.bar(y=col, rot=30)
			title_insert = " ({0})".format(strategy_function_names[col].lower())
			filename_insert = "_{0}".format(col)
			title = prepend + "Satisfied applicants{insert}: {header}".format(insert=title_insert, header=header)
			filename = filename_pre + "{insert}_satisfied_{header}".format(insert=filename_insert, header=header)
			self._plot("Applicants who got their nth preference", "%" if percent else "count", title, filename)
	def plot_all_stratified(self, percent=True, filter_f=None, prepend="", filename_pre=""):
		"""Separate function for plotting everything from a stratified dataframe"""
		# TODO: fix bug, positions on chart are out of order
		if filter_f != None:
			result_df = self.stratify_and_filter_results(filter_f)
		else:
			result_df = self.stratified_results
		# consider additionally using a 3d plot to show each xs
		result_df = self.percentify_stratified_results(result_df) if percent else result_df
		for f in self.strategies_used:
			toplot = result_df.xs(f)
			# print(toplot.to_string())
			toplot.plot.bar(rot=30)
			title_insert = " ({0})".format(strategy_function_names[f].lower())
			filename_insert = "_{0}".format(f)
			self._plot("Applicants who got their nth preference", "%" if percent else "count",
				prepend + "Satisfied applicants{insert}".format(insert=title_insert),
				filename_pre + "{insert}_satisfied".format(insert=filename_insert))
	def plot_every_stratified(self, percent=True, filter_f=None, prepend="", filename_pre=""):
		"""Separate function for plotting each column from a stratified dataframe"""
		for col in self.results:
			self.plot_one_stratified(col, percent, filter_f, prepend, filename_pre)
	@staticmethod
	def _plot(xlab, ylab, title, filename=""):
		if not filename:
			filename = title
		plt.xlabel(xlab)
		plt.ylabel(ylab)
		plt.title('\n'.join(wrap(title, 60)))
		plt.tight_layout()
		plt.savefig("images/"+sanitise_filename(filename)+".png", dpi=300)
		# plt.show()
		plt.clf()
		plt.cla()
		plt.close('all')
	def current_unhappiness(self):
		return sum(a.unhappiness() for a in self.applicants)
	@staticmethod
	def expand(group):
		a = []
		for i in range(len(group)):
			a += [i]*group[i]
		return a
	def compare_two_subgroups(self, cols, filter_f=None, category=0):
		"""Run Mann-Whitney U test between two specified subgroups
		Also allows for filters and composite subgroups i.e. adding >1 subgroup into one
		Should exclude unallocated applicants by default
		Both cols should be passed as lists
		stats.mannwhitneyu(a,b)"""
		# raise an error if there are any duplicates
		result_df = self._recut_results("cat{0}".format(category+1), False, filter_f).iloc[:15]
		print(result_df)
		flattened_cols = [c for sublist in cols for c in sublist]
		if len(uniq(flattened_cols)) < len(flattened_cols):
			raise Exception
		col_a, col_b = cols
		# do some cross-sectional magic
		group_a = result_df[col_a].sum(axis=1)
		group_b = result_df[col_b].sum(axis=1)
		# expand the group columns at this point
		expand_a = self.expand(group_a)
		expand_b = self.expand(group_b)
		print(cols)
		print(np.mean(expand_a), np.mean(expand_b))
		return stats.mannwhitneyu(expand_a, expand_b)
	def compare_all_subgroups(self, filter_f=None):
		"""Runs Kruskal-Wallis H test between all subgroups
		stats.kruskal(a,b,c)"""
		toplot = self._recut_results("cat1", False, filter_f).iloc[:15]
		print(toplot)
		# expand the group columns at this point
		expands = []
		for col in toplot:
			expand = []
			group = toplot[col]
			for i in range(len(group)):
				expand += [i]*group[i]
			expands.append(expand)
		print([np.mean(ex) for ex in expands])
		return stats.kruskal(*expands)
	def compare_two_firsts(self, cols, filter_f=None, category=0):
		result_df = self._recut_results("cat{0}".format(category+1), True, filter_f).iloc[:15]
		print(result_df)
		flattened_cols = [c for sublist in cols for c in sublist]
		if len(uniq(flattened_cols)) < len(flattened_cols):
			raise Exception
		col_a, col_b = cols
		# do some cross-sectional magic
		group_a = result_df[col_a].sum(axis=1)
		group_b = result_df[col_b].sum(axis=1)
		group_a = group_a/group_a.sum()
		group_b = group_b/group_b.sum()
		row_a = [group_a[0], sum(group_a[1:])]
		row_b = [group_b[0], sum(group_b[1:])]
		chi2_table = np.array([row_a, row_b]).T
		print(cols)
		print(chi2_table)
		return stats.chi2_contingency(chi2_table)

# Numba optimisations
# TODO: optimise these methods with CUDA/Python for GPU usage
# https://github.com/Hellisotherpeople/Simulated-Annealing-TSP-Numba/blob/master/simulated_annealing_tsp_numba_real.py

@numba.jit(fastmath=True,nopython=True)
def accept(energy, new_energy, T):
	if new_energy < energy:
		return 1
	else:
		return np.exp((energy - new_energy)/T)

@numba.jit(fastmath=True,nopython=True)
def swap_two(lst):
	new_list = lst.copy()
	app_len = len(lst)-1
	a, b = np.random.choice(app_len, 2)
	temp = new_list[b]
	new_list[b] = new_list[a]
	new_list[a] = temp
	return new_list

@numba.jit(fastmath=True,nopython=True)
def cool_gpu(current_state_arr, pref_arr, capacity_arr, T, cool_rate, iterlimit):
	temp = T
	itercount = 0
	unhappiness_log = np.empty((0,2),np.int64)
	min_unhappiness = POSITIVE_INFINITY
	while temp >= 1e-8 and itercount < iterlimit:
		next_state_arr = swap_two(current_state_arr)
		u_current = np.sum(np.multiply(pref_arr, current_state_arr))
		u_next = np.sum(np.multiply(pref_arr, next_state_arr))
		next_over_capacity = (np.sum(next_state_arr,axis=0) > capacity_arr).any()
		# if itercount < 10:
		# 	print(current_state_arr, next_state_arr, pref_arr)
		# 	print(np.sum(next_state_arr,axis=0), capacity_arr)
		# 	print(u_current, u_next)
		if accept(u_current, u_next, temp) >= np.random.random() and not next_over_capacity:
			current_state_arr = next_state_arr
			u_current = u_next
		if u_current < min_unhappiness:
			best_state_arr = current_state_arr
			min_unhappiness = u_current
		temp *= 1 - cool_rate
		itercount += 1
		unhappiness_log = np.append(unhappiness_log, np.array([[u_current, min_unhappiness]],np.int64), axis=0)
	return (temp, min_unhappiness, current_state_arr, best_state_arr, unhappiness_log)

@numba.jit(fastmath=True,nopython=True)
def step_gpu(current_state_arr, pref_arr, iterlimit, starting_unhappiness):
	itercount = 0
	unhappiness_log = np.empty((0,2),np.int64)
	min_unhappiness = starting_unhappiness
	while itercount < iterlimit:
		next_state_arr = swap_two(current_state_arr)
		u_current, u_next = [np.sum(np.multiply(pref_arr, s)) for s in [current_state_arr, next_state_arr]]
		if accept(u_current, u_next, temp) >= np.random.random():
			current_state_arr = next_state_arr
			u_current = u_next
		if u_current < min_unhappiness:
			best_state_arr = current_state_arr
			min_unhappiness = u_current
		itercount += 1
		unhappiness_log = np.append(unhappiness_log, np.array([[u_current, min_unhappiness]],np.int64), axis=0)
	return (min_unhappiness, current_state_arr, best_state_arr, unhappiness_log)

# Simulated annealing based simulation

class AnnealSimulation(Simulation):
	"""Simulation that uses Simulated Annealing (as outlined in the official HETI document)
	
	GPU functionality does not work completely. The intent was to run the cool_gpu and step_gpu
	functions via numba.cuda.jit but I can't figure out how to use cuda so I'm just going to leave it
	with numpy arrays for now. It's still significantly faster than self._step_cool()"""
	def __init__(self, starting_strategy, gpu=True, temp=10000, cool_rate=0.003, iterlimit=1000000):
		self.min_unhappiness = POSITIVE_INFINITY
		self._use_gpu = gpu # ok I know it says "gpu" but it's really just numba.jit, not cuda
		self.best_state = None
		self.current_state = None
		self.best_state_arr = np.array([])
		self.pref_arr = np.array([])
		self.capacity_arr = np.array([])
		self.current_state_arr = np.array([])
		self.temp = temp
		self.cooling_rate = cool_rate
		self.iterlimit = iterlimit
		self.unhappiness_records = pd.DataFrame(columns=["current_unhappiness", "min_unhappiness"])
		self.unhappiness_array = np.empty((0,2),dtype=int)
		super(AnnealSimulation, self).__init__(starting_strategy)
	def initial_state(self):
		for category in range(len(self.category_counts)):
			if category == 0:
				for hospital in self.hospitals:
					hospital.fill(self.unplaced(self.applicants, category))
			else:
				leftover_applicants = self.unplaced(self.applicants, category)
				leftover_spots = sum(h.spots_remaining for h in self.hospitals)
				# print("{apps} applicants remain to be placed in {spots} spots".format(apps=len(leftover_applicants),
				# 	spots=leftover_spots))
				if leftover_spots < len(leftover_applicants):
					random.shuffle(leftover_applicants)
				for hospital in self.hospitals:
					hospital.fill(leftover_applicants[:leftover_spots])
		self.current_state = self.placed(self.applicants)
		self.min_unhappiness = min(self.min_unhappiness, self.current_unhappiness())
		return self.current_state
	def _gpu_translate(self):
		self.current_state_arr = np.zeros((len(self.current_state), len(hospitals)),dtype=int)
		for a in range(len(self.current_state)):
			self.current_state_arr[a,self.hospitals.index(self.current_state[a].allocation)] = 1
		self.pref_arr = np.array([[a.preferences.index(b) for b in self.hospitals] for a in self.current_state])
		self.capacity_arr = np.array([h.spots_remaining for h in self.hospitals])
		self.best_state_arr = self.current_state_arr
	def _gpu_detranslate(self):
		# bug: hospitals do not allocate
		append_data = pd.DataFrame({"current_unhappiness": list(self.unhappiness_array[:,0]),
			"min_unhappiness": list(self.unhappiness_array[:,1])})
		self.unhappiness_records = self.unhappiness_records.append(append_data, ignore_index=True)
		# self.unhappiness_records.plot.line()
		# plt.show()
		
		for hospital in self.hospitals:
			hospital.empty()
		for a in range(len(self.current_state_arr)):
			current_one = np.where(self.current_state_arr[a] == 1)[0][0]
			# do stuff
			self.current_state[a].allocate(self.hospitals[current_one])
		if (self.best_state_arr == self.current_state_arr).all():
			self.best_state = self.current_state
		self._update_applicants()
		if self.min_unhappiness != self.current_unhappiness():
			raise Exception("Calculated unhappiness of this arrangement is {min_uh} but actual is {cur_uh}".format(
				min_uh=self.min_unhappiness,cur_uh=self.current_unhappiness()))
		self._make_results()
	def _runsim(self):
		"""Runs simulated annealing procedures"""
		self._dra_prefill()
		self.current_state = self.initial_state()
		self.best_state = deepcopy(self.current_state)
		self.unhappiness_records = self.unhappiness_records.append(
			{"current_unhappiness": self.current_unhappiness(),
			"min_unhappiness": self.min_unhappiness}, ignore_index=True)
		if not self._use_gpu:
			self._step_cool()
		else:
			self._gpu_translate()
			self._step_cool_gpu()
	def _step_cool(self):
		"""Object-oriented code
		Code based on http://www.theprojectspot.com/tutorial-post/simulated-annealing-algorithm-for-beginners/6
		
		PAINFULLY SLOW, takes several minutes to do 10000 iterations"""
		itercount = 0
		while self.temp >= 1e-8 and itercount < self.iterlimit:
			next_state = deepcopy(self.current_state)
			self.swap(next_state)
			u_current, u_next = [self.unhappiness(s) for s in [self.current_state, next_state]]
			if self.accept(u_current, u_next) >= random.random():
				self.current_state = next_state
			if self.current_unhappiness() < self.min_unhappiness:
				self.best_state = self.current_state
				self.min_unhappiness = self.current_unhappiness()
			self.temp *= 1 - self.cooling_rate
			itercount += 1
			self.unhappiness_records = self.unhappiness_records.append(
				{"current_unhappiness": self.current_unhappiness(),
				"min_unhappiness": self.min_unhappiness}, ignore_index=True)
		# self.unhappiness_records.plot.line()
		# plt.show()
		self._update_applicants()
		self._make_results()
	def _step_cool_gpu(self):
		"""Meant to have converted everything to an array representation so it can be gpu-optimised.
		
		VERY FAST, does 30000 iterations in a few seconds"""
		temp, min_unhappiness, current_state_arr, best_state_arr, unhappiness_log = cool_gpu(self.current_state_arr,
			self.pref_arr, self.capacity_arr, self.temp, self.cooling_rate, self.iterlimit)
		self.temp = temp
		self.min_unhappiness = min_unhappiness
		self.current_state_arr = current_state_arr
		self.best_state_arr = best_state_arr
		self.unhappiness_array = np.append(self.unhappiness_array, unhappiness_log, axis=0)
		self._gpu_detranslate()
	def step(self, iters):
		"""Mainly for testing in the console - iterate the process a specified number of times to see
		if it makes any difference to global unhappiness."""
		itercount = 0
		while itercount < iters:
			next_state = deepcopy(self.current_state)
			AnnealSimulation.swap(next_state)
			u_current, u_next = list(map(AnnealSimulation.unhappiness, [self.current_state, next_state]))
			if self.accept(u_current, u_next) >= random.random():
				self.current_state = next_state
			if self.current_unhappiness() < self.min_unhappiness:
				self.best_state = self.current_state
				self.min_unhappiness = self.current_unhappiness()
			itercount += 1
			self.unhappiness_records = self.unhappiness_records.append(
				{"current_unhappiness": self.current_unhappiness(),
				"min_unhappiness": self.min_unhappiness}, ignore_index=True)
		self.unhappiness_records.plot.line()
		plt.show()
		self._update_applicants()
		self._make_results()
	def step_gpu(self, iters):
		"""Calls the numba function step_gpu"""
		# cbf implementing this shit, final year exams are a bitch
		raise NotImplementedError
	def _update_applicants(self):
		self.applicants = self.current_state + self.unplaced(self.applicants)
	@staticmethod
	def unhappiness(state):
		"""Cost function for an allocation state"""
		return sum(a.unhappiness() for a in state)
	@staticmethod
	def swap(state):
		"""Swap two within an allocation state"""
		a, b = random.sample(state, 2)
		a.swap(b)
	def current_unhappiness(self):
		return self.unhappiness(self.current_state)
	def accept(self, energy, new_energy):
		if new_energy < energy:
			return 1
		else:
			return exp((energy - new_energy)/self.temp)

# TODO: create a new class to run the arithmogeometric and demon annealing methods described in here
# the demon method description is not entirely correct, wikipedia does a better job
# http://www.scielo.org.mx/pdf/cys/v21n3/1405-5546-cys-21-03-00493.pdf
# there doesn't seem to be any particular advantage to doing a million operations
# if you plot this function, you'd need a very long integer to store the number of decimal places
# required to get n=6, and a very small value of alpha.
# n = log10(8+log10(T0)) - log10(-log10(1-a))
# there are far more efficient ways of reaching the same outcome without the unnecessary wasteful use
# of computational resources, which I will try to implement in the next update
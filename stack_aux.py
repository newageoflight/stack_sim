#!/usr/bin/env python3

from copy import deepcopy
from matplotlib.ticker import PercentFormatter
from math import exp
from numpy.random import choice
from textwrap import wrap

import matplotlib.pyplot as plt
import math
import numba
import numpy as np
import pandas as pd
import random
import re

plt.style.use("ggplot")
plt.rcParams["font.family"] = "Avenir Next"

# Constants

POSITIVE_INFINITY = 1e8
NEGATIVE_INFINITY = -POSITIVE_INFINITY

category_counts = []
with open("category-counts.txt", "r") as categories_infile:
	for line in categories_infile:
		catid, catnum = line.split('\t')
		category_counts.append(int(catnum))

# Basic functions

ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])
sanitise_filename = lambda x: re.sub(r'[<>:"/\|?*]', '', x)

# Classes

class Hospital(object):
	"""Hospitals are assumed to have one property that counts in the algorithm:
	- Capacity
	Further pathways like DRA can also be implemented"""
	def __init__(self, name: str, abbreviation: str, capacity: int, firsts: int, dra: int, remove_dra=False):
		self.name = name
		self.abbreviation = abbreviation
		self.is_dra = dra > 0
		self.dra = dra
		self.capacity = capacity if not remove_dra else capacity - dra
		self.firsts = firsts
		self.spots_remaining = capacity
		self.filled_spots = []
	def __repr__(self):
		return self.__str__()
	def __str__(self):
		return "'{name}' ({abbr}): {filled}/{capacity}".format(name=self.name, abbr=self.abbreviation, filled=self.capacity - self.spots_remaining, capacity=self.capacity)
	def fill(self, applicants, dra_prefill=False):
		if len(applicants) > self.spots_remaining:
			selected_applicants = random.sample(applicants, self.spots_remaining)
		else:
			selected_applicants = applicants
		self.filled_spots += selected_applicants
		self.spots_remaining -= len(selected_applicants)
		for a in self.filled_spots:
			a.allocate(self, dra_prefill)
	def empty(self):
		while self.filled_spots:
			applicant = self.filled_spots.pop()
			applicant.free()
			self.spots_remaining += 1

# Class-dependent constants

hospitals = []
with open("hospital-networks.txt", "r") as hospital_infile:
	for line in hospital_infile:
		hname, hshort, hcap, hfirsts, hdra = line.split('\t')
		hospitals.append(Hospital(hname, hshort, int(hcap), int(hfirsts), int(hdra)))
hospital_weights = [h.firsts/366 for h in hospitals]
hospitals_with_weights = list(zip(hospitals, hospital_weights))

stack = []
with open("stack.txt", "r") as stack_infile:
	for line in stack_infile:
		stack.append(next((h for h in hospitals if h.abbreviation == line.strip()), None))
stack_weights = [h.firsts/366 for h in stack]

altstack = []
with open("altstack.txt", "r") as altstack_infile:
	for line in altstack_infile:
		altstack.append(next((h for h in hospitals if h.abbreviation == line.strip()), None))

# stack strategy functions
# these need to be fixed somehow so they don't look retarded like they do now (global variables, redundant arguments)
# may need e.g. a separate class for HospitalList

def shuffle(l):
	return random.sample(l, len(l))

def weighted_shuffle(l,w=hospital_weights):
	return list(choice(l, len(l), p=w, replace=False))

def push_random_to_top(l):
	k = l
	k.insert(0, k.pop(random.randint(0,len(k)-1)))
	return k

def push_wt_random_to_top(l,w=stack_weights):
	k = l
	k.insert(0, k.pop(choice(len(k), 1, p=w)[0]))
	return k

def push_wt_random_to_position(l,n,w=stack_weights):
	k = l
	k.insert(n, k.pop(choice(len(k), 1, p=w)[0]))
	return k

def default_stack(l):
	global stack
	return stack

def mixed_stacks(l):
	global stack, altstack
	return random.choice([stack, altstack])

def push_random_to_top_and_n(l,n):
	# randomly select two and then place at positions 0 and n-1
	return push_random_to_positions(l, 0, n)
	
def push_wt_random_to_top_and_n(l,n,w=stack_weights):
	# weighted-randomly select two and then place at positions 0 and n-1
	return push_wt_random_to_positions(l, 0, n)

def push_random_to_positions(l, *positions):
	k = l
	pairs = zip(positions, choice(len(k), len(positions)))
	for target, origin in pairs:
		k.insert(target, k.pop(origin))
	return k

def push_wt_random_to_positions(l, *positions, w=stack_weights):
	k = l
	pairs = zip(positions, choice(len(k), len(positions), p=w))
	for target, origin in pairs:
		k.insert(target, k.pop(origin))
	return k

class Applicant(object):
	"""An applicant is assumed to have two properties that count in the algorithm:
	- Order of preferences
	- Category"""
	def __init__(self, strategy, category, reject=0):
		self.strategy = strategy.__name__
		self.preferences = strategy(hospitals.copy())
		self.category = category
		self.allocation = None
		self.preference_number = None
		self.is_dra = False
		self.reject_pr = reject
	def __repr__(self):
		return self.__str__()
	def __str__(self):
		return "Category {cat} applicant allocated to {alloc} ({prefn}): {prefs}".format(cat=self.category+1,
			prefs=[h.abbreviation for h in self.preferences], alloc=self.allocation.abbreviation if self.allocation else "NONE", prefn=self.preference_number)
	def allocate(self, hospital, dra_prefill=False):
		reject = random.random() < self.reject_pr
		if not reject:
			self.allocation = hospital
			self.is_dra = dra_prefill
			self.preference_number = self.preferences.index(self.allocation)
	def swap(self, other):
		temp = other.allocation
		other.allocate(self.allocation)
		self.allocate(temp)
	def free(self):
		self.allocation = None
		self.preference_number = None
	def unhappiness(self):
		return 0 if not self.preference_number else self.preference_number
	
# Simulation objects

class Simulation(object):
	"""Runs a simulation of the allocation process"""
	def __init__(self, starting_strategy, dra_prefill=False, rounds=1):
		self.category_counts = category_counts
		self.hospitals = hospitals.copy()
		for hospital in self.hospitals:
			hospital.empty()
		self.applicants = [Applicant(starting_strategy, cat) for cat in range(len(category_counts)) for i in range(category_counts[cat])]
		self.allocation_rounds = rounds
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
		for category in range(len(self.category_counts)):
			for hospital in self.hospitals:
				for rank in range(len(self.hospitals)):
					preferenced_this = [a for a in self.unplaced() if a.preferences[rank] == hospital and a.category == category]
					hospital.fill(preferenced_this)
		self._make_results()
	def satisfied(self, rank, category=None):
		if category != None:
			return [a for a in self.applicants if a.preference_number == rank and a.category == category]
		else:
			return [a for a in self.applicants if a.preference_number == rank]
	def placed(self, category=None):
		if category != None:
			return [a for a in self.applicants if a.allocation!=None and a.category==category]
		else:
			return [a for a in self.applicants if a.allocation!=None]
	def unplaced(self, category=None):
		if category != None:
			return [a for a in self.applicants if a.allocation==None and a.category==category]
		else:
			return [a for a in self.applicants if a.allocation==None]
	def dra_only(self):
		return [a for a in self.applicants if a.is_dra]
	def non_dra_only(self):
		return [a for a in self.applicants if not a.is_dra]
	def _make_results(self):
		panda_d = {}
		placed = len(self.placed())
		not_placed = len(self.unplaced())
		total = sum(self.category_counts)
		panda_d["total"] = [len(self.satisfied(rank)) for rank in range(len(self.hospitals))] + [placed, not_placed, total]
		for category in range(len(self.category_counts)):
			cat_placed = len(self.placed(category))
			cat_not_placed = len(self.unplaced(category))
			cat_total = self.category_counts[category]
			panda_d["cat{cat}".format(cat=category+1)] = [len(self.satisfied(rank, category)) for rank in range(len(self.hospitals))] + [cat_placed, cat_not_placed, cat_total]
		self.results = pd.DataFrame(panda_d, index=[ordinal(n) for n in range(1, len(self.hospitals)+1)]+["placed", "not_placed", "total"])
		return self.results
	def percentify_results(self):
		new_results = self.results.copy()
		for col in new_results:
			new_results[col] = 100*new_results[col]/new_results[col]["total"]
		return new_results.iloc[:17]
	def export_results(self, name: str):
		self.results.to_csv("tables/"+name+".csv")
		self.percentify_results().to_csv("tables/"+name+"_percentified.csv")
	def pprint(self):
		"""Legacy function: pretty-prints out the results"""
		for rank in range(len(self.hospitals)):
			satisfied = self.satisfied(rank)
			print("Total applicants who got their {ord} preference: {count} ({percent:.2%})".format(ord=ordinal(rank+1), count=len(satisfied), 
				percent=len(satisfied)/sum(self.category_counts)))
		placed = len(self.placed())
		not_placed = len(self.unplaced())
		total = sum(self.category_counts)
		print("Total applicants who received any placement: {count} ({percent:.2%})".format(count=placed, percent=placed/total))
		print("Total applicants who did not get any placement: {count} ({percent:.2%})".format(count=not_placed, percent=not_placed/total))
		print("Total applicants: {count}".format(count=sum(self.category_counts), percent=placed/total))
		for category in range(len(self.category_counts)):
			for rank in range(len(self.hospitals)):
				satisfied = self.satisfied(rank, category)
				print("Total Category {cat} applicants who got their {ord} preference: {count} ({percent:.2%})".format(ord=ordinal(rank+1), count=len(satisfied), 
					percent=len(satisfied)/category_counts[category], cat=category+1))
			cat_placed = len(self.placed(category))
			cat_not_placed = len(self.unplaced(category))
			cat_total = self.category_counts[category]
			print("Total Category {cat} applicants who received any placement: {count} ({percent:.2%})".format(cat=category+1, count=cat_placed, percent=cat_placed/cat_total))
			print("Total Category {cat} applicants who did not get any placement: {count} ({percent:.2%})".format(cat=category+1, count=cat_not_placed, percent=cat_not_placed/cat_total))
			print("Total Category {cat} applicants: {count}".format(cat=category+1, count=cat_total))
	def plot_one(self, header, percent=True, prepend="", filename_pre=""):
		toplot = self.percentify_results() if percent else self.results
		fig, ax = plt.subplots()
		ax.yaxis.set_major_formatter(PercentFormatter())
		title = prepend + "Satisfied applicants: {header}".format(header=header)
		filename = filename_pre + "_satisfied_{header}".format(header=header)
		toplot.plot.bar(y=header, rot=30)
		self._plot("Applicants who got their nth preference", "%" if percent else "count", title, filename)
	def plot_all(self, percent=True, prepend="", filename_pre=""):
		toplot = self.percentify_results() if percent else self.results
		toplot.plot.bar(rot=30)
		self._plot("Applicants who got their nth preference", "%" if percent else "count", prepend + "Satisfied applicants",
			filename_pre + "_satisfied")
	def plot_every(self, percent=True, prepend="", filename_pre=""):
		for col in self.results:
			self.plot_one(col, percent, prepend, filename_pre)
	def _plot(self, xlab, ylab, title, filename=""):
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
	def __init__(self, starting_strategy, gpu=True, temp=10000, cool_rate=0.0002, iterlimit=1000000):
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
					hospital.fill(self.unplaced(category))
			else:
				leftover_applicants = self.unplaced(category)
				leftover_spots = sum(h.spots_remaining for h in self.hospitals)
				# print("{apps} applicants remain to be placed in {spots} spots".format(apps=len(leftover_applicants),
				# 	spots=leftover_spots))
				if leftover_spots < len(leftover_applicants):
					random.shuffle(leftover_applicants)
				for hospital in self.hospitals:
					hospital.fill(leftover_applicants[:leftover_spots])
		self.current_state = self.placed()
		self.min_unhappiness = min(self.min_unhappiness, self.current_unhappiness())
		return self.current_state
	def _gpu_translate(self):
		self.current_state_arr = np.zeros((len(self.current_state), len(hospitals)),dtype=int)
		for a in range(len(self.current_state)):
			self.current_state_arr[a,self.hospitals.index(self.current_state[a].allocation)] = 1
		self.pref_arr = np.array([[a.preferences.index(b) for b in self.hospitals] for a in self.current_state])
		self.capacity_arr = np.array([h.capacity for h in self.hospitals])
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
			best_one = np.where(self.best_state_arr[a] == 1)[0][0]
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
			AnnealSimulation.swap(next_state)
			u_current, u_next = [AnnealSimulation.unhappiness(s) for s in [self.current_state, next_state]]
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
		raise NotImplementedError
	def _update_applicants(self):
		self.applicants = self.current_state + self.unplaced()
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
		return AnnealSimulation.unhappiness(self.current_state)
	def accept(self, energy, new_energy):
		if new_energy < energy:
			return 1
		else:
			return exp((energy - new_energy)/self.temp)

# Test object(s)

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

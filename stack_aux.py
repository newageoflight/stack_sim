#!/usr/bin/env python3

from matplotlib.ticker import PercentFormatter
from math import exp
from copy import deepcopy
from numpy.random import choice

import matplotlib.pyplot as plt
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

# basic functions

ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])
sanitise_filename = lambda x: re.sub(r'[<>:"/\|?*]', '', x)

# Classes

class Hospital(object):
	"""Hospitals are assumed to have one property that counts in the algorithm:
	- Capacity"""
	def __init__(self, name: str, abbreviation: str, capacity: int, firsts: int, dra: int, remove_dra=False):
		self.name = name
		self.abbreviation = abbreviation
		self.dra = dra
		self.capacity = capacity if not remove_dra else capacity - dra
		self.firsts = firsts
		self.spots_remaining = capacity
		self.filled_spots = []
	def __repr__(self):
		return self.__str__()
	def __str__(self):
		return "'{name}' ({abbr}): {filled}/{capacity}".format(name=self.name, abbr=self.abbreviation, filled=self.capacity - self.spots_remaining, capacity=self.capacity)
	def fill(self, applicants):
		if len(applicants) > self.spots_remaining:
			selected_applicants = random.sample(applicants, self.spots_remaining)
		else:
			selected_applicants = applicants
		self.filled_spots += selected_applicants
		self.spots_remaining -= len(selected_applicants)
		for a in self.filled_spots:
			a.allocate(self)
	def empty(self):
		while self.filled_spots:
			applicant = self.filled_spots.pop()
			applicant.free()
			self.spots_remaining += 1

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

# stack strategy functions
# these need to be fixed somehow so they don't look retarded like they do now (global variables, redundant arguments)
# may need e.g. a separate class for HospitalList

def shuffle(l):
	return random.sample(l, len(l))

def weighted_shuffle(l):
	global hospital_weights
	return list(choice(l, len(l), p=hospital_weights, replace=False))

def push_random_to_top(l):
	k = l
	k.insert(0, k.pop(random.randint(0,len(k)-1)))
	return k

def push_wt_random_to_top(l):
	global hospital_weights
	k = l
	k.insert(0, k.pop(choice(len(k), 1, p=hospital_weights)[0]))
	return k

def default_stack(l):
	global stack
	return stack

# def push_random_to_top_and_14(l):
# 	k = push_random_to_top(l)
# 	k.insert(13, k.pop(random.randint(1,len(k)-1)))
# 	return k

# def push_wt_random_to_top_and_14(l):
# 	global hospitals_with_weights
# 	k = push_random_to_top(hospitals_with_weights)
# 	k_weights = [i[1] for i in k]
# 	k.insert(13, k.pop(list(choice(list(range(1,len(k))), 1, p=k_weights))))
# 	return [i[0] for i in k]

class Applicant(object):
	"""An applicant is assumed to have two properties that count in the algorithm:
	- Order of preferences
	- Category"""
	def __init__(self, strategy, category):
		self.strategy = strategy.__name__
		self.preferences = strategy(hospitals[:])
		self.category = category
		self.allocation = None
		self.preference_number = None
	def __repr__(self):
		return self.__str__()
	def __str__(self):
		return "Category {cat} applicant allocated to {alloc} ({prefn}): {prefs}".format(cat=self.category+1,
			prefs=[h.abbreviation for h in self.preferences], alloc=self.allocation.abbreviation, prefn=self.preference_number)
	def allocate(self, hospital):
		self.allocation = hospital
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
	def __init__(self, starting_strategy):
		self.category_counts = category_counts
		self.hospitals = hospitals[:]
		for hospital in self.hospitals:
			hospital.empty()
		self.applicants = [Applicant(starting_strategy, cat) for cat in range(len(category_counts)) for i in range(category_counts[cat])]
		self.results = pd.DataFrame()
		self._runsim()
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
	def plot_one(self, header, percent=True, prepend=""):
		toplot = self.percentify_results() if percent else self.results
		fig, ax = plt.subplots()
		ax.yaxis.set_major_formatter(PercentFormatter())
		title = prepend + "Satisfied applicants: {header}".format(header=header)
		toplot.plot.bar(y=header, rot=30)
		self._plot("Applicants who got their nth preference", "%" if percent else "count", title)
	def plot_all(self, percent=True, prepend=""):
		toplot = self.percentify_results() if percent else self.results
		toplot.plot.bar(rot=30)
		self._plot("Applicants who got their nth preference", "%" if percent else "count", prepend + "Satisfied applicants")
	def plot_every(self, percent=True, prepend=""):
		for col in self.results:
			self.plot_one(col, percent, prepend)
	def _plot(self, xlab, ylab, title):
		plt.xlabel(xlab)
		plt.ylabel(ylab)
		plt.title(title)
		plt.tight_layout()
		plt.savefig("images/"+sanitise_filename(title)+".png", dpi=300)
		# plt.show()
		plt.clf()
		plt.cla()
		plt.close('all')
	def current_unhappiness(self):
		return sum(a.unhappiness() for a in self.applicants)

@numba.jit
def accept(energy, new_energy, T):
	if new_energy < energy:
		return 1
	else:
		return np.exp((energy - new_energy)/T)

@numba.jit
def cool_gpu(current_state_arr, pref_arr, T, cool_rate, iterlimit):
	temp = T
	itercount = 0
	unhappiness_log = np.empty((0,2),dtype=int)
	min_unhappiness = POSITIVE_INFINITY
	while temp >= 1e-8 and itercount < iterlimit:
		next_state_arr = current_state_arr
		# have to define the swap locally otherwise GPU functions don't work
		i,j = choice(len(next_state_arr), 2)
		next_state_arr[[i,j]] = next_state_arr[[j,i]]
		u_current, u_next = [np.sum(np.multiply(pref_arr, s)) for s in [current_state_arr, next_state_arr]]
		if accept(u_current, u_next, T) >= np.random.random():
			current_state_arr = next_state_arr
			u_current = u_next
		if u_current < min_unhappiness:
			best_state_arr = current_state_arr
			min_unhappiness = u_current
		temp *= 1 - cool_rate
		itercount += 1
		unhappiness_log = np.append(unhappiness_log, [[u_current, min_unhappiness]], axis=0)
	return (temp, min_unhappiness, current_state_arr, best_state_arr, unhappiness_log)

class AnnealSimulation(Simulation):
	"""Simulation that uses Simulated Annealing (as outlined in the official HETI document)

	TODO: Implement GPU/Numpy related speed optimisations. Really bugged at the moment"""
	def __init__(self, starting_strategy, gpu=False):
		self.min_unhappiness = POSITIVE_INFINITY
		self._use_gpu = gpu
		self.best_state = None
		self.current_state = None
		self.best_state_arr = np.array([])
		self.pref_arr = np.array([])
		self.current_state_arr = np.array([])
		self.temp = 100000
		self.cooling_rate = 0.001
		self.iterlimit = 1000000
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
		self.current_state = list(self.placed())
		self.min_unhappiness = self.unhappiness(self.current_state)
		return self.current_state
	def _gpu_translate(self):
		self.current_state_arr = np.zeros((len(self.current_state), len(hospitals)),dtype=int)
		for a in range(len(self.current_state)):
			self.current_state_arr[a,self.hospitals.index(self.current_state[a].preferences[0])] = 1
		self.pref_arr = np.array([[self.hospitals.index(b) for b in a.preferences] for a in self.current_state])
		self.best_state_arr = self.current_state_arr
	def _gpu_detranslate(self):
		print(self.min_unhappiness)
		print(self.current_state_arr)
		self.unhappiness_records = self.unhappiness_records.append(
			pd.DataFrame({"current_unhappiness": list(self.unhappiness_array[:,0]),
			"min_unhappiness": list(self.unhappiness_array[:,1])}), ignore_index=True)
		print(self.unhappiness_records.to_string())
		self.unhappiness_records.plot.line()
		plt.show()
		
		for a in range(len(self.current_state_arr)):
			the_one = np.where(self.current_state_arr[a] == 1)[0][0]
			self.current_state[a].free()
			self.current_state[a].allocate(self.hospitals[the_one])
			self.best_state[a].free()
			self.best_state[a].allocate(self.hospitals[the_one])
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
		Painfully slow, takes like a minute to do 10000 iterations"""
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
	# @numba.jit
	def _step_cool_gpu(self):
		"""Meant to have converted everything to an array representation so it can be gpu-optimised.
		numba doesn't work with class methods so this will need to be fixed somehow
		If this works, it will take <1 second to do the same job that the other function does in 30"""
		temp, min_unhappiness, current_state_arr, best_state_arr, unhappiness_log = cool_gpu(self.current_state_arr,
			self.pref_arr, self.temp, self.cooling_rate, self.iterlimit)
		# itercount = 0
		# while self.temp >= 1e-8 and itercount < self.iterlimit:
		# 	next_state_arr = self.current_state_arr
		# 	# have to define the swap locally otherwise GPU functions don't work
		# 	i,j = choice(len(next_state_arr), 2)
		# 	next_state_arr[[i,j]] = next_state_arr[[j,i]]
		# 	u_current, u_next = [np.sum(np.multiply(self.pref_arr, s)) for s in [self.current_state_arr, next_state_arr]]
		# 	if self.accept(u_current, u_next) >= np.random.random():
		# 		self.current_state_arr = next_state_arr
		# 		u_current = u_next
		# 	if u_current < self.min_unhappiness:
		# 		self.best_state_arr = self.current_state_arr
		# 		self.min_unhappiness = u_current
		# 	self.temp *= 1 - self.cooling_rate
		# 	itercount += 1
		self.temp = temp
		self.min_unhappiness = min_unhappiness
		self.current_state_arr = current_state_arr
		self.best_state_arr = best_state_arr
		self.unhappiness_array = np.append(self.unhappiness_array, unhappiness_log, axis=0)
		self._gpu_detranslate()
	def step(self, iters):
		"""Mainly for testing in the console - iterate the process a specified number of times to see
		if it makes any difference."""
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
		raise NotImplementedError
		itercount = 0
		while itercount < iters:
			next_state_arr = self.current_state_arr
			AnnealSimulation.swap_arr(next_state_arr)
			u_current, u_next = [np.sum(np.multiply(self.pref_arr, s)) for s in [self.current_state_arr, next_state_arr]]
			if self.accept(u_current, u_next) >= np.random.random():
				self.current_state_arr = next_state_arr
				u_current = u_next
			if u_current < self.min_unhappiness:
				self.best_state_arr = self.current_state_arr
				self.min_unhappiness = u_current
			self.temp *= 1 - self.cooling_rate
			itercount += 1
			self.unhappiness_array.append(np.array([u_current, self.min_unhappiness]), axis=0)
		self.gpu_detranslate()
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

# given that this is a minimisation problem, other approaches like GD can also be used
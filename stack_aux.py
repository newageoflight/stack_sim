#!/usr/bin/env python3

from matplotlib.ticker import PercentFormatter

import matplotlib.pyplot as plt
import pandas as pd
import random
import re

plt.style.use("ggplot")
plt.rcParams["font.family"] = "Avenir Next"

# Constants

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
	def __init__(self, name: str, abbreviation: str, capacity: int):
		self.name = name
		self.abbreviation = abbreviation
		self.capacity = capacity
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
		hname, hshort, hcap = line.split('\t')
		hospitals.append(Hospital(hname, hshort, int(hcap)))

stack = []
with open("stack.txt", "r") as stack_infile:
	for line in stack_infile:
		stack.append(next((h for h in hospitals if h.abbreviation == line.strip()), None))

# stack strategy functions

def shuffle(l):
	return random.sample(l, len(l))

def push_random_to_top(l):
	k = l[:]
	k.insert(0, k.pop(random.randint(0,len(k)-1)))
	return k

def default_stack(l):
	global stack
	return stack

def push_random_to_top_and_14(l):
	k = push_random_to_top(l)
	k.insert(13, k.pop(random.randint(1,len(k)-1)))
	return k

class Applicant(object):
	"""An applicant is assumed to have two properties that count in the algorithm:
	- Order of preferences
	- Category"""
	def __init__(self, strategy, category: int):
		self.strategy = strategy.__name__
		self.preferences = strategy(hospitals[:])
		self.category = category
		self.allocation = None
		self.preference_number = None
	def __repr__(self):
		return self.__str__()
	def __str__(self):
		return "Category {cat} applicant: {prefs}; allocated to {alloc} ({prefn})".format(cat=self.category+1,
			prefs=[h.abbreviation for h in self.preferences], alloc=self.allocation.abbreviation, prefn=self.preference_number)
	def allocate(self, hospital: Hospital):
		self.allocation = hospital
		self.preference_number = self.preferences.index(self.allocation)
	def free(self):
		self.allocation = None
		self.preference_number = None

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
					preferenced_this = list(filter(lambda a: a.preferences[rank] == hospital and a.category == category, [a for a in self.applicants if not a.allocation]))
					hospital.fill(preferenced_this)
		self._make_results()
	def satisfied(self, rank: int, category=None):
		if category != None:
			return filter(lambda a: a.preference_number == rank and a.category == category, self.applicants)
		else:
			return filter(lambda a: a.preference_number == rank, self.applicants)
	def placed(self, category=None):
		if category != None:
			return filter(lambda a: a.allocation!=None and a.category==category, self.applicants)
		else:
			return filter(lambda a: a.allocation!=None, self.applicants)
	def unplaced(self, category=None):
		if category != None:
			return filter(lambda a: a.allocation==None and a.category==category, self.applicants)
		else:
			return filter(lambda a: a.allocation==None, self.applicants)
	def _make_results(self):
		panda_d = {}
		placed = len(list(self.placed()))
		not_placed = len(list(self.unplaced()))
		total = sum(self.category_counts)
		panda_d["total"] = [len(list(self.satisfied(rank))) for rank in range(len(self.hospitals))] + [placed, not_placed, total]
		for category in range(len(self.category_counts)):
			cat_placed = len(list(self.placed(category)))
			cat_not_placed = len(list(self.unplaced(category)))
			cat_total = self.category_counts[category]
			panda_d["cat{cat}".format(cat=category+1)] = [len(list(self.satisfied(rank, category))) for rank in range(len(self.hospitals))] + [cat_placed, cat_not_placed, cat_total]
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
			satisfied = list(self.satisfied(rank))
			print("Total applicants who got their {ord} preference: {count} ({percent:.2%})".format(ord=ordinal(rank+1), count=len(satisfied), 
				percent=len(list(satisfied))/sum(self.category_counts)))
		placed = len(list(self.placed()))
		not_placed = len(list(self.unplaced()))
		total = sum(self.category_counts)
		print("Total applicants who received any placement: {count} ({percent:.2%})".format(count=placed, percent=placed/total))
		print("Total applicants who did not get any placement: {count} ({percent:.2%})".format(count=not_placed, percent=not_placed/total))
		print("Total applicants: {count}".format(count=sum(self.category_counts), percent=placed/total))
		for category in range(len(self.category_counts)):
			for rank in range(len(self.hospitals)):
				satisfied = list(self.satisfied(rank, category))
				print("Total Category {cat} applicants who got their {ord} preference: {count} ({percent:.2%})".format(ord=ordinal(rank+1), count=len(satisfied), 
					percent=len(satisfied)/category_counts[category], cat=category+1))
			cat_placed = len(list(self.placed(category)))
			cat_not_placed = len(list(self.unplaced(category)))
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
		plt.close()

class AnnealerSimulation(Simulation):
	"""Simulation that uses Simulated Annealling (as outlined in the official HETI document"""
	def __init__(self, starting_strategy):
		super(AnnealerSimulation, self).__init__(starting_strategy)
	def _runsim(self):
		"""Planning to use scipy or similar library to maximise global_happiness"""
		raise NotImplementedError
	def global_happiness(self):
		return sum(a.preference_number for a in self.applicants)
#!/usr/bin/env python3

"""
Base functions and classes
"""

from numpy.random import choice
from functools import wraps

import random
import re

# Constants

POSITIVE_INFINITY = 1e8
NEGATIVE_INFINITY = -POSITIVE_INFINITY

category_counts = []
with open("category-counts.txt", "r") as categories_infile:
	for line in categories_infile:
		catid, catnum = line.split('\t')
		category_counts.append(int(catnum))

# Basic functions

ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
sanitise_filename = lambda x: re.sub(r'[<>:"/\|?*,]', '', x)
def uniq(l):
	seen = []
	for i in l:
		if i not in seen:
			seen.append(i)
	return seen
def underscorify(name):
	new_name = re.sub(r"[\s]", "_", name)
	new_name = sanitise_filename(new_name.lower())
	return new_name

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
		return "'{name}' ({abbr}): {filled}/{capacity}".format(name=self.name, abbr=self.abbreviation,
	   		filled=self.capacity - self.spots_remaining, capacity=self.capacity)
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
stack_with_weights = list(zip(hospitals, hospital_weights))
tier_one_hospitals = stack[:4]
top_six_hospitals = stack[:6]

altstack = []
with open("altstack.txt", "r") as altstack_infile:
	for line in altstack_infile:
		altstack.append(next((h for h in hospitals if h.abbreviation == line.strip()), None))

# stack strategy functions
# these need to be fixed somehow so they don't look retarded like they do now (global variables, redundant arguments)
# may need e.g. a separate class for HospitalList

# here's an idea that may fix it
# have a callable object that contains a preferencing strategy
# when called (takes an argument), it can check if it was initiated with a list or a function and return accordingly

class StrategyObj(object):
	"""Callable object containing a preferencing strategy
	Can either be a list of hospitals or a function of that list"""
	def __init__(self, name, strategy):
		self.name = name
		self.strategy = strategy
		self.__name__ = strategy.__name__ if callable(strategy) else underscorify(name)
	def __call__(self, *args, **kwargs):	
		if callable(self.strategy):
			return self.strategy(*args, **kwargs)
		elif type(self.strategy) == list:
			return self.strategy

def Strategy(function=None, name=""):
	if function:
		return StrategyObj(name, function)
	else:
		@wraps(function)
		def wrapper(function):
			return StrategyObj(name, function)
		return wrapper

# strategies

@Strategy(name="Random")
def shuffle(l):
	return random.sample(l[:], len(l))

@Strategy(name="Weighted random")
def weighted_shuffle(l,w=hospital_weights):
	return list(choice(l[:], len(l), p=w, replace=False))

def push_random_to_top(l):
	k = l[:]
	k.insert(0, k.pop(random.randint(0,len(k)-1)))
	return k

@Strategy(name="Stack with random top")
def stack_random_top(l):
	global stack
	return push_random_to_top(stack)

def push_wt_random_to_top(l,w=hospital_weights):
	k = l[:]
	k.insert(0, k.pop(choice(len(k), 1, p=w)[0]))
	return k

@Strategy(name="Random with weighted random top")
def random_with_wt_random_top(l,w=hospital_weights):
	global hospitals_with_weights
	k = hospitals_with_weights[:]
	random.shuffle(k)
	lst, wts = zip(*k)
	return push_wt_random_to_top(list(lst), list(wts))

@Strategy(name="Stack with weighted random top")
def stack_wt_random_top(l):
	global stack
	return push_wt_random_to_top(stack, w=stack_weights)

def push_wt_random_to_position(l,n,w=hospital_weights):
	k = l
	k.insert(n, k.pop(choice(len(k), 1, p=w)[0]))
	return k

# default_stack = StrategyObj("Stack", stack.copy())
@Strategy(name="Stack")
def default_stack(l):
	global stack
	return stack.copy()

stack_rearr_cache = []
def stack_rearrange():
	# This function has since been updated to reflect that several order swap proposals exist
	# They are mainly based around hearsay and differ vastly between universities
	# Therefore I will assume that the distribution is random
	# This is treated as a separate function from the default stack so the default case is excluded
	global stack_with_weights, stack_rearr_cache
	base_stack = stack_with_weights.copy()
	# return random.choice([stack.copy(), altstack.copy()])
	# proposals involve swapping one or more pairs of adjacent hospitals.
	# for the sake of this simulation there are a few rules:
	# 1. pairs cannot overlap (i.e. cannot choose [2,3] and [3,4] to swap)
	# 2. the maximum allowable swap count is 5 - unrealistic that there would be more
	# therefore...
	# choose a random number between 1 and 5 first
	pair_count = 3
	# create a list of pairs to swap
	indices_remaining = set(range(len(base_stack)))
	pairs = []
	# iteratively generate pairs
	for i in range(pair_count):
		pair_first = random.choice(tuple(indices_remaining))
		pair_second = pair_first + 1 if pair_first != len(base_stack) - 1 else pair_first - 1
		pair = set([pair_first, pair_second])
		indices_remaining -= pair
		pairs.append(pair)
	# now swap the pairs
	stack_rearr_cache.append(pairs)
	# for pair in pairs:
	# 	i,j = pair
	# 	temp = base_stack[i]
	# 	base_stack[i] = base_stack[j]
	# 	base_stack[j] = temp
	# # return the rearranged stack
	# return base_stack

for i in range(7):
	stack_rearrange()

def stack_rearrangement(l, w=hospital_weights):
	global stack_rearr_cache, stack_with_weights
	base_stack = stack_with_weights.copy()
	pairs = random.choice(stack_rearr_cache)
	pairs = random.sample(pairs, random.randint(1,3))
	for pair in pairs:
		i,j = pair
		temp = base_stack[i]
		base_stack[i] = base_stack[j]
		base_stack[j] = temp
	# return the rearranged stack
	return base_stack

@Strategy(name="Mixed stacks")
def mixed_stacks(l):
	lst, wts = zip(*stack_rearrangement(l))
	return list(lst)

@Strategy(name="Mixed stacks with weighted random top")
def mixed_stack_wt_random_top(l, w=hospital_weights):
	lst, wts = zip(*stack_rearrangement(l, w))
	return push_wt_random_to_top(list(lst), list(wts))

def push_random_to_top_and_n(l,n):
	# randomly select two and then place at positions 0 and n-1
	return push_random_to_positions(l, 0, n)
	
def push_wt_random_to_top_and_n(l,n,w=stack_weights):
	# weighted-randomly select two and then place at positions 0 and n-1
	return push_wt_random_to_positions(l, 0, n, w=w)

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

# TODO: obsolete this dictionary, rely on names given in StrategyObj class
strategy_function_names = {
	"": "",
	"shuffle": "Random",
	"weighted_shuffle": "Weighted random",
	"default_stack": "Stack",
	"mixed_stacks": "Mixed stacks",
	"stack_random_top": "Stack with random top",
	"stack_wt_random_top": "Stack with weighted random top",
	"mixed_stack_wt_random_top": "Stack with weighted random top",
	"random_with_wt_random_top": "Random with weighted random top",
}

# Filters
# Should this be represented as an object as well?

class FilterObj(object):
	"""Callable object containing a filter"""
	def __init__(self, name, strategy):
		self.name = name
		self.strategy = strategy
		self.__name__ = strategy.__name__ if callable(strategy) else underscorify(name)
	def __call__(self, *args, **kwargs):
		if callable(self.strategy):
			return self.strategy(*args, **kwargs)
		elif type(self.strategy) == list:
			return self.strategy

def Filter(function=None, name=""):
	if function:
		return StrategyObj(name, function)
	else:
		@wraps(function)
		def wrapper(function):
			return StrategyObj(name, function)
		return wrapper

# functions

def wanted_top_4_hospital(applicant):
	global tier_one_hospitals
	return applicant.preferences[0] in tier_one_hospitals

def got_top_4_hospital(applicant):
	global tier_one_hospitals
	return applicant.allocation in tier_one_hospitals

def wanted_top_6_hospital(applicant):
	global top_six_hospitals
	return applicant.preferences[0] in top_six_hospitals

def got_top_6_hospital(applicant):
	global top_six_hospitals
	return applicant.allocation in top_six_hospitals

filter_function_names = {
	"wanted_top_4_hospital": "Wanted a top 4 hospital",
	"got_top_4_hospital": "Got a top 4 hospital",
	"wanted_top_6_hospital": "Wanted a top 6 hospital",
	"got_top_6_hospital": "Got a top 6 hospital",
}

# More base classes

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
			prefs=[h.abbreviation for h in self.preferences],
			 alloc=self.allocation.abbreviation if self.allocation else "NONE", prefn=self.preference_number)
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

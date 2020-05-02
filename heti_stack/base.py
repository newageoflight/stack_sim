#!/usr/bin/env python3

"""
Base functions and classes
"""

from numpy.random import choice

import math
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

ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])
sanitise_filename = lambda x: re.sub(r'[<>:"/\|?*]', '', x)
def uniq(l):
	seen = []
	for i in l:
		if i not in seen:
			seen.append(i)
	return seen

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

# Mixed strategies

def mixed_strategy_stack_and_random():
	return random.choice([default_stack, weighted_shuffle])

def mixed_strategy_stack_and_wt_random():
	return choice([default_stack, weighted_shuffle], 1, p=[0.8, 0.2])[0]

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

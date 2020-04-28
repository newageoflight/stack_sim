#!/usr/bin/env python3

# TODO: update this using matplotlib and pandas/numpy instead of this shit

import random

ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])

class Applicant(object):
	"""An applicant is assumed to have two properties that count in the algorithm:
	- Order of preferences
	- Category"""
	def __init__(self, preferences, category):
		self.preferences = preferences
		self.category = category
		self.allocation = None
		self.preference_number = None
	def __repr__(self):
		return self.__str__()
	def __str__(self):
		return "Category {cat} applicant: {prefs}; allocated to {alloc} ({prefn})".format(cat=self.category+1,
			prefs=[h.abbreviation for h in self.preferences], alloc=self.allocation.abbreviation, prefn=self.preference_number)
	def allocate(self, hospital):
		self.allocation = hospital
		self.preference_number = self.preferences.index(self.allocation)
		# print("A category {cat} applicant has been located to {hosp}, which was their {num} preference".format(
		# 	cat=self.category+1, hosp=self.allocation.abbreviation, num=ordinal(self.preference_number+1)))

class Hospital(object):
	"""Hospitals are assumed to have one property that counts in the algorithm:
	- Capacity"""
	def __init__(self, name, abbreviation, capacity):
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

hospitals = []
with open("hospital-networks.txt", "r") as hospital_infile:
	for line in hospital_infile:
		hname, hshort, hcap = line.split('\t')
		hospitals.append(Hospital(hname, hshort, int(hcap)))

stack = []
with open("stack.txt", "r") as stack_infile:
	for line in stack_infile:
		stack.append(next((h for h in hospitals if h.abbreviation == line.strip()), None))

category_counts = []
with open("category-counts.txt", "r") as categories_infile:
	for line in categories_infile:
		catid, catnum = line.split('\t')
		category_counts.append(int(catnum))

def shuffle(l):
	return random.sample(l, len(l))

def push_random_to_top(l):
	k = l[:]
	k.insert(0, k.pop(random.randint(0,len(k)-1)))
	return k

def default_stack(l):
	global stack
	return stack

# Assuming all preferences are random
# applicants_all_random = [Applicant(random.sample(hospitals, len(hospitals)), cat) for i in range(cat) for cat in category_counts]
# Assuming everyone uses the stack
# applicants_all_stacked = [Applicant(stack, cat) for i in range(cat) for cat in category_counts]
# Randomly selecting applicants to use the stack but move a random hospital to the top
# applicants_stacked_with_random_first = [Applicant(push_random_to_top(stack), cat) for i in range(cat) for cat in category_counts]

class Simulation(object):
	"""docstring for Simulation"""
	def __init__(self, starting_function):
		self.category_counts = category_counts
		self.hospitals = hospitals[:]
		self.applicants = [Applicant(starting_function(self.hospitals), cat) for cat in range(len(category_counts)) for i in range(category_counts[cat])]
		self.runsim()
	def runsim(self):
		for category in range(len(self.category_counts)):
			for hospital in self.hospitals:
				for rank in range(len(self.hospitals)):
					preferenced_this = list(filter(lambda a: a.preferences[rank] == hospital and a.category == category, [a for a in self.applicants if not a.allocation]))
					hospital.fill(preferenced_this)
		self.pprint()
	def change_function(self, new_function):
		self.applicants = [Applicant(new_function(self.hospitals), cat) for cat in range(len(category_counts)) for i in range(category_counts[cat])]
	def pprint(self):
		print(sum(self.category_counts), self.category_counts)
		for rank in range(len(self.hospitals)):
			satisfied = list(filter(lambda a: a.preference_number == rank, self.applicants))
			print("Total applicants who got their {ord} preference: {count} ({percent:.2%})".format(ord=ordinal(rank+1), count=len(satisfied), 
				percent=len(list(satisfied))/sum(self.category_counts)))
		print("Total applicants who received any placement: {count}".format(count=len(list(filter(lambda a: a.allocation!=None, self.applicants)))))
		print("Total applicants who did not get any placement: {count}".format(count=len(list(filter(lambda a: a.allocation==None, self.applicants)))))
		print("Total applicants: {count}".format(count=sum(self.category_counts))
		for category in range(len(self.category_counts)):
			for rank in range(len(self.hospitals)):
				satisfied = list(filter(lambda a: a.preference_number == rank and a.category == category, self.applicants))
				print("Total Category {cat} applicants who got their {ord} preference: {count} ({percent:.2%})".format(ord=ordinal(rank+1), count=len(satisfied), 
					percent=len(satisfied)/category_counts[category], cat=category+1))
			print("Total Category {cat} applicants who received any placement: {count}".format(cat=category+1, count=len(list(filter(lambda a: a.allocation!=None and a.category==category, self.applicants)))))
			print("Total Category {cat} applicants who did not get any placement: {count}".format(cat=category+1, count=len(list(filter(lambda a: a.allocation==None and a.category==category, self.applicants)))))
			print("Total Category {cat} applicants: {count}".format(cat=category+1, count=self.category_counts[category]))

# bug: sims will not run separately, all 3 instances affect the same "hospitals" list

# print("""
# If all applicants select completely randomly:
# ------------------
# """)
# applicants_all_random = Simulation(shuffle)

# print("""
# If all applicants stack:
# ------------------
# """)
# applicants_all_stacked = Simulation(default_stack)

print("""
If all applicants stack but move a random hospital to first:
------------------
""")
applicants_stacked_with_random_first = Simulation(push_random_to_top)
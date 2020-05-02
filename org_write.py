#!/usr/bin/env python3

import csv

def org_table_from_csv(csv_file):
	table_string = ""
	line_count = 0
	with open(csv_file, "r") as csv_in:
		reader = csv.reader(csv_in)
		for row in reader:
			table_string += "|" + "|".join(row) + "|\n"
			if line_count == 0:
				table_string += "|-\n"
			line_count += 1
	return table_string

def write_results(tests, unhappiness, org_file):
	with open(org_file, "w") as outfile:
		print("* Results", file=outfile)
		weighted_tests = [k for k in tests if "weighted" in k.name.lower()]
		unweighted_tests = [k for k in tests if "weighted" not in k.name.lower()]
		test_segregations = {
			"weighted": weighted_tests,
			"unweighted": unweighted_tests
		}
		for k, d in test_segregations.items():
			print("** AI algorithm + {0} random selection".format(k), file=outfile)
			for test in d:
				print("*** {0}".format(test.name), file=outfile)
				print("[[./images/{0}_satisfied.png]]".format(test.underscore_name), file=outfile)
				print(file=outfile)
				print(org_table_from_csv("./tables/{0}.csv".format(test.underscore_name)),file=outfile)
				for i in range(3):
					if i == 0:
						print("**** Total", file=outfile)
						print("[[./images/{0}_satisfied_total.png]]".format(test.underscore_name), file=outfile)
					else:
						print("**** Category {0}".format(i), file=outfile)
						print("[[./images/{0}_satisfied_cat{1}.png]]".format(test.underscore_name, i), file=outfile)
		print("** AI algorithm convergence", file=outfile)
		for test in tests:
			print("*** {0}".format(test.name), file=outfile)
			print("[[./images/conv_{0}.png]]".format(test.underscore_name), file=outfile)
		print("** Global unhappiness when compared to categorical matching", file=outfile)
		print(org_table_from_csv("./tables/{0}.csv".format(unhappiness)),file=outfile)
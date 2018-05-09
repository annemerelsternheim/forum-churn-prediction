import re
data = ['abc_12', 'abc_15', 'abc_21', 'abc_31', 'abc_15', 'abc_71', 'abc_12', 'abc_11', 'abc_01']

seen_users=[re.search("_([0-9]*)", filename).group(1) for filename in data]

print seen_users

users = ['12','15','21','31','15','71','12','11','01']

for user in users:
	if user in seen_users:
		next
	else:
		print "blablabla"

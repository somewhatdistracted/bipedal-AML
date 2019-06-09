import csv

csv_copy_right = []
csv_copy_left = []
	
with open('allsubject.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		right_row = []
		left_row = []
		for count, element in enumerate(row):
			right_row.append(float(element))
			left_row.append(float(element))
			if count == 2:
				right_row[count] = float(element) * -1
			if count == 5:
				left_row[count] = float(element) * -1
		csv_copy_right.append(right_row)
		csv_copy_left.append(left_row)

final_csv = []
for i in range(24):
	for j in range(202):
		if j < 50 :
			v = j
			final_csv.append(csv_copy_left[v + 101*i])
		elif j < 151 :
			v = j
			if v > 100:
				v = v - 101
			final_csv.append(csv_copy_right[v + 101*i])
		else :
			v = j
			if v > 101:
				v = v - 101
			final_csv.append(csv_copy_left[v + 101*i])

direction_csv = []
for row in final_csv:
	new_row = []
	for element in row:
		new_row.append(float(element))
	direction_csv.append(new_row)
	
for i in range(24):
	for j in range(202):
		if j > 0:
			for element in final_csv[(j-1) + 202*i]:
				direction_csv[j + 202*i].append(element)
		else:
			for element in final_csv[201 + 202*i]:
				direction_csv[j + 202*i].append(element)
		
		print(j + 202*i)	
			
with open('newSubjectDataDirection.csv', mode='w') as new_file:
		file_writer = csv.writer(new_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		
		for row in direction_csv :
			file_writer.writerow(row)
		
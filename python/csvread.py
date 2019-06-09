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
			print(v + 101*i)
			final_csv.append(csv_copy_left[v + 101*i])
		elif j < 151 :
			v = j
			if v > 100:
				v = v - 101
			print(v + 101*i)
			final_csv.append(csv_copy_right[v + 101*i])
		else :
			v = j
			if v > 101:
				v = v - 101
			print(v + 101*i)
			final_csv.append(csv_copy_left[v + 101*i])

target_csv = []
for i in range(24):
	for j in range(202):
		if j < 201:
			target_csv.append(final_csv[(j+1) + i*101])
		else:
			target_csv.append(final_csv[i*101])
			
with open('newSubjectData.csv', mode='w') as new_file:
		file_writer = csv.writer(new_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		
		for row in final_csv :
			file_writer.writerow(row)

with open('newSubjectDataResults.csv', mode='w') as new_file:
		file_writer = csv.writer(new_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		
		for row in target_csv :
			file_writer.writerow(row)
		
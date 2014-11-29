import pandas as pd

def convert_to_sf(raw_value):
	key_words=['met','m','M']
	#first convert any multiplicative items
	if (raw_value.find('x')!=-1):
		#print 'working on ' + raw_value
		nums=raw_value.split('x')
		num1=float(nums[0])
		rest=nums[1].split()
		num2=float(rest[0])
		square_f=num1*num2
		print raw_value
		if (len(rest)==1):
			rest.append('SF')
		print raw_value + ' converted to ' + str(square_f) + ' '+ rest[1]
		raw_value=str(square_f) + ' ' + rest[1]

	#everything is in the squared format. put meters to feet
	words = raw_value.split()
	SF=float(words[0].replace(",",""))
	for val in key_words:
		if (raw_value.find(val)!=-1):
			#convert to squared feet
			SF =SF*10.7639
			break

	print raw_value + ' converted to ' + str(SF)
	return SF

data = pd.read_csv('extra_data_v2.csv')
#print data
#extract 'Area' field as well and convert to squared feet
i=0
for row in data.values:
	
	row[1]=convert_to_sf(str(row[1]))
	row[6]=convert_to_sf(str(row[6]))
	#data['Area'][i]=row[17]
	#print 'this is item ' + str(data['LivingArea'][i])
	data['LivingArea'][i]=row[1]
	data['Area'][i]=row[6]
	i=i+1
	

data.to_csv('post_extra_data.csv')


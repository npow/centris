import math
import pandas as pd 

def distance_on_unit_sphere(lat1, long1, lat2, long2):
 
 	#source code at http://www.johndcook.com/blog/python_longitude_latitude/
    # Convert latitude and longitude to 
    # spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0
         
    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
         
    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
         
    # Compute spherical distance from spherical coordinates.
         
    # For two locations in spherical coordinates 
    # (1, theta, phi) and (1, theta, phi)
    # cosine( arc length ) = 
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length
     
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) + 
           math.cos(phi1)*math.cos(phi2))
    arc = math.acos( cos )
 
    # Remember to multiply arc by the radius of the earth 
    # in your favorite set of units to get length.
    return arc*6373

def find_closest_point(point,list_points):
	distance=[]
	for point2 in list_points:
		distance.append(distance_on_unit_sphere(point[0],point[1],point2[0],point2[1]))
	#print distance
	return min(distance)

#load all the firestation coordinates into a list

# with open('firestations.csv') as f:
# 	f.readline()
# 	for line in f:
# 		line=line.split(',')
# 		fire.append((float(line[1]),float(line[2])))
# print fire
# print fire[0][0]

fireData=pd.read_csv('firestations.csv')
fire_lati=fireData['lat']
fire_long=fireData['lng']
fire=zip(fire_lati,fire_long)

#load all the police station coordinates into a list

policeData=pd.read_csv('policeCoord.csv')
police_lati=policeData['lat']
police_long=policeData['long']
police = zip(police_lati,police_long)

listings=pd.read_csv('modified_listings.csv')
house_location=zip(listings['Lat'],listings['Lng'])
# print type(house_location)
# print house_location[0][1]
police_list=[]
fire_list=[]
for i,point in enumerate(house_location):
	closest_police=find_closest_point(point,police)
	closest_fire=find_closest_point(point,fire)
	police_list.append(closest_police)
	fire_list.append(closest_fire)

# print max(fire_list)
# print max(police_list)
# print min(fire_list)
# print min(police_list)
# print sum(fire_list)/len(fire_list)
# print sum(police_list)/len(police_list)
listings['PoliceDist'] = police_list
listings['FireDist']=fire_list
listings.to_csv('modified_listings_2.csv')




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


def weight_vector(lat1,long1,points,sigma):
	weights=[]
	for point in points:

		distance=distance_on_unit_sphere(lat1,long1,point[0],point[1])
		weight=gaussian_weight(distance,sigma)
		weights.append(weight)
		print "House at distance ",str(distance)," has weight ",str(weight)
	return weights

def gaussian_weight(distance,sigma):
	return math.exp(-distance**2/(2*sigma**2))

#print gaussian_weight(0.1,1.7)
listings=pd.read_csv('modified_listings.csv')
points=zip(listings['Lat'],listings['Lng'])
lat1=points[0][0]
long1=points[0][1]
#print weight_vector(lat1,long1,points[1:20],1)
weights=weight_vector(lat1,long1,points[1:40],1.647)
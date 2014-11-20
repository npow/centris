def locate_city(lat,lon,cities,coord):
    for i in range(len(cities)):
        if found_city(lat,lon,coord[i]):
            print 'We found the point in city: ' + cities[i]
            break
            
def found_city(lat,lon,coord):
    cn=0    
    for i in range(len(coord)-1):
        p1=coord[i].split()
        p2=coord[i+1].split()
        p1y=float(p1[0])
        p1x=float(p1[1])
        p2y=float(p2[0])
        p2x=float(p2[1])
        if ((p1y<=lon)&(p2y>lon))|((p1y>lon)&(p2y<=lon)):
            vt = float(lon-p1y)/float(p2y-p1y)
            if (lat<p1x+vt*(p2x-p1x)):
                cn+=1
    #print 'Number of crossings ' + str(cn) + '\n'
    return cn%2
        
city=[]
coord=[]
with open("BoundingBox.xml") as f:
    for line in f:
        #check if it is the name of the borough
        if 'NOM' in line:
            #print line
            line=line[line.find('>')+1:line.find('/')-1]
            #print line
            city.append( line)
        if 'MULTIPOLYGON' in line:
            cord = line[line.find('(')+3:line.find(')')-1]            
            coord.append(cord.split(','))            
            #print cord

locate_city(45.486617, -73.822392,city,coord)

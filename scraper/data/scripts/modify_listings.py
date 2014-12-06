# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 11:49:52 2014

@author: eemilja
"""

import csv
import extractCord
import numpy as np

cities=dict()
genrecode=dict()
catgcode=dict()
GenreCode=['AP','C','ME','VE','I','4X','MA','PP','AU','MPM','TE','2X','TR','3X','MEM','LS','MM','5X','FE']
CatgCode=['COP','PCI','UNI','PPR','TER','FER']
output_lines=[]
with open('listings_v2.csv','rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    header=next(reader,None) 
    #create new header based on one-hot encoding of City, Genre Code, and CatgCode
    city,coord=get_city_coord()
    count=-1    
    for c in city:
        count+=1
        cities[c]=count
    count=-1;
    for g in GenreCode:
        count+=1
        genrecode[g]=count
    count=-1;
    for c in CatgCode:
        count+=1
        catgcode[c]=count
    header=header[0:3]+header[4:6]+header[8:]+city+GenreCode+CatgCode
    #print city
    #print cities.keys()
    output_lines.append(header)
    output_file=open('modified_listings.csv',"wb")
    writer = csv.writer(output_file, delimiter=',')
    writer.writerow(header)
    for input in reader:
         #intialize all empty zero vector for city, genrecode, and catgcode
        cityVec=np.zeros(len(city),dtype=np.int8)
        catgVec=np.zeros(len(CatgCode),dtype=np.int8)
        genreVec=np.zeros(len(GenreCode),dtype=np.int8)
        line=input
        lat=float(line[1])
        lon=float(line[2])
        current_city=locate_city(lat,lon,city,coord)
        #print current_city
        cityVec[cities[current_city]]=1
        #print cities_new.values()
        genreVec[genrecode[line[6]]]=1
        catgVec[catgcode[line[7]]]=1
        line=line[0:3]+line[4:6]+line[8:]+cityVec.astype('str').tolist()+genreVec.astype('str').tolist()+catgVec.astype('str').tolist()
        output_lines.append(line)
        writer.writerow(line)
    print 'we go all outputs'
    output_file.close()
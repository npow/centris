import geocoder
import pandas as pd
import sys

INPUT_FILE = '../hist_DUPROPRIO_v2.csv'

data = pd.read_csv(INPUT_FILE)
addresses = data[['Address']].values
f = open('geocoded_listings.csv', 'wb')
f.write('id,address,lat,lng\n')
for i, address in enumerate(addresses):
    address = address[0]
    print address
    sys.stdout.flush()
    c = geocoder.google(address)
    if c.status.find('ERROR') > -1:
        f.write('%d,"%s",,\n' % (i, address))
    else:
        f.write('%d,"%s",%f,%f\n' % (i, address, c.lat, c.lng))
    f.flush()
f.close()

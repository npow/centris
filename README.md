# Requirements

## Classifier

* [anaconda](http://continuum.io/downloads)

##### Dependencies
```
# conda update scikit-learn numpy
# pip install pybrain pandas geocoder
```

## Scraper

* [nodejs](http://nodejs.org/)

##### Dependencies
```
# npm install -g bluebird request json2csv fast-csv jsdom
```

Note: You may need to set `NODE_PATH=/usr/local/lib/node_modules`
##### Run scraper. Responses are saved under `responses`
```
node fetch_listings.js
```

##### Combine responses. This creates a file `data/listings.csv`
```
node combine.js
```

##### Fetch additional data for a `provider` (eg. `REMAX`, `SUTTON`, `C21`). Responses are saved under `extra_data/$PROVIDER`
```
# touch ${PROVIDER}_errors.txt
# node fetch_details.js $PROVIDER
```

##### Firestation data
Run `notebooks/firestations.ipynb`

##### Fetch historical data 
```
# ./fetch_hist_DUPROPRIO.sh
# node extract_DUPROPRIO.js
# node combine.js
```

##### Geocode historical listings. This creates a file `data/scripts/geocoded_listings.csv`
```
# python geocode_listings.py
```

### Requirements

* [nodejs](http://nodejs.org/)

### Dependencies
```
# npm install -g bluebird request json2csv
```

Note: You may need to set `NODE_PATH=/usr/local/lib/node_modules`
### Run scraper. Responses are saved under `responses`
```
node fetch_listings.js
```

### Combine responses. This creates a file `data/listings.csv`
```
node combine.js
```

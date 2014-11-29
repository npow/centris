var Promise = require('bluebird');
var csv = Promise.promisifyAll(require('fast-csv'));
var fs = Promise.promisifyAll(require('fs'));
var jsdom = Promise.promisifyAll(require('jsdom'));
var json2csv = require('json2csv');
var sys = Promise.promisifyAll(require('sys'));
var readdirp = Promise.promisifyAll(require('readdirp'));
var request = Promise.promisifyAll(require('request'));

var INDEX = process.argv[2];
var ERROR_FILE = 'DUPROPRIO_errors.txt';

function fetchDUPROPRIO(url, data) {
  return jsdom.envAsync(url, ["http://code.jquery.com/jquery.js"])
    .then(function (window) {
      var $ = window.$;
      // missing: FloorCovering, Area, Insurance, Topography
      var fields = {
        'Asking Price :': 'AskingPrice',
        'Year of construction :': 'YearBuilt',
        'Lot dimensions :': 'Area',
        'Number of bathrooms :': 'NumberBathrooms',
        'Number of rooms :': 'NumberRooms',
        'Number of bedrooms :': 'NumberBedrooms',
        'Number of levels': 'NumberLevels',
        'Located on which floor?': 'FloorNumber',
        'Living space area': 'LivingArea',
        'Municipal Assessment :': 'MunicipalAssessment',
        'Number of interior parking :': 'Parking',
        'Number of exterior parking :': 'Parking'
      };
      for (var field in fields) {
        var key = fields[field];
        var res = $('li:contains("' + field + '")');
        if (res && res.length > 0) {
          var value = res.first().text().trim();
          value = value.substring(value.indexOf(':')+1).trim();
          if (value.length > 0) {
            data[key] = value;
          }
        }
      }
      console.log(data);
      fs.writeFileSync('tmp/DUPROPRIO/hist/json' + data.id + '.json', JSON.stringify(data));

      window.close();
  })
  .error(function () {
    fs.appendFileSync(ERROR_FILE, url+'\n');
    return Promise.resolve();
  });
}

function processFile(fileName) {
  var html = fs.readFileSync(fileName).toString();
  return jsdom.envAsync(html, ["http://code.jquery.com/jquery.js"])
    .then(function (window) {
      var $ = window.$;
      return Promise.all([].map.call($('.resultData'), function (x) {
        var id = $(x).find('a').attr('href');
        var url = 'http://duproprio.com' + id;
        var saleDate = $(x).first().find('p:contains("Sold")').text();
        var priceSold = $(x).first().find('p:contains("Price sold")').text().trim();
        priceSold = priceSold.substring(priceSold.indexOf(':')+1);
        var askingPrice = $(x).first().find('p:contains("Asking price")').text().trim();
        askingPrice = askingPrice.substring(askingPrice.indexOf(':')+1);
        if (fs.existsSync('tmp/DUPROPRIO/hist/json' + id + '.json')) {
          console.log('Skipping: ' + id);
          return Promise.resolve();
        }
        var data = { AskingPrice: askingPrice, PriceSold: priceSold, SaleDate: saleDate, id: id };
        return fetchDUPROPRIO(url, data);
      }));
    })
    .error(function () {
      console.log('Error processing: ' + fileName);
    });
}

var L = [];
readdirp({ root: 'tmp/DUPROPRIO/hist/html', fileFilter: INDEX + '.html' })
  .on('data', function (entry) {
    L.push(processFile(entry.fullPath));
  })
  .on('end', function () {
    Promise.all(L).then(function () {
      console.log('DONE');
    });
  });

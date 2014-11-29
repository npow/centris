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
      var fields = {
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
        var address = $($(x).children()[0].children[1]).text().trim().split('\n').map(function (x) {
          return x.trim();
        }).join('\n');
        var category = $(x).first().find('a').first().text().trim().split(',')[0]
        var saleDate = $(x).first().find('p:contains("Sold")').text().trim();
        var priceSold = $(x).first().find('p:contains("Price sold")').text().trim();
        priceSold = priceSold.substring(priceSold.indexOf(':')+1).trim();
        var askingPrice = $(x).first().find('p:contains("Asking price")').text().trim();
        askingPrice = askingPrice.substring(askingPrice.indexOf(':')+1).trim();
        var info = $(x).first().find('.resultMeta').text().trim();
        var data = {
          id: id,
          Address: address,
          AskingPrice: askingPrice,
          Info: info,
          PriceSold: priceSold,
          SaleDate: saleDate,
          Category: category
        };
        var targetFileName = 'tmp/DUPROPRIO/hist/json' + id + '.json';
        if (fs.existsSync(targetFileName)) {
          /*
          var data2 = JSON.parse(fs.readFileSync(targetFileName).toString());
          for (var key in data) {
            var value = data[key];
            if (value.length > 0) {
              data2[key] = value;
            }
          }
          fs.writeFileSync('tmp/DUPROPRIO/hist/json/new' + id + '.json', JSON.stringify(data));
          */
          console.log('Skipping: ' + id);
          return Promise.resolve();
        }
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

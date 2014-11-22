var Promise = require('bluebird');
var child_process = Promise.promisifyAll(require('child_process'));
var fs = Promise.promisifyAll(require('fs'));
var json2csv = require('json2csv');
var readdirp = Promise.promisifyAll(require('readdirp'));

combineExtra();

function combine() {
  var L = [];
  for (var i = 1; i <= 3668; ++i) {
    data = JSON.parse(fs.readFileSync('responses/' + i + '.json').toString())
    L = L.concat(data.d.Result);
  }
  console.log(L.length);
  json2csv({ data: L, fields: Object.keys(L[0]).filter(function (x) { return ['Icons', 'Zonages', 'Brokers'].indexOf(x) === -1; }) }, function(err, csv) {
    if (err) console.log(err);
    fs.writeFile('data/listings.csv', csv, function(err) {
      if (err) throw err;
      console.log('DONE');
    });
  });
}

function combineExtra() {
  var fileList = [];
  var keys = {};
  readdirp({ root: 'extra_data', fileFilter: '*.json' })
    .on('data', function (entry) {
      fileList.push(entry.fullPath);
    })
    .on('end', function () {
      Promise.all(fileList.map(function (x) {
        return fs.readFileAsync(x).then(function (data) {
          return JSON.parse(data);
        });
      }))
      .then(function (L) {
        var keys = {};
        L.forEach(function (x) {
          for (var key in x) {
            keys[key] = 1;
          }
        });
        json2csv({ data: L, fields: Object.keys(keys) }, function (err, csv) {
          if (err) console.log(err);
          fs.writeFile('data/extra_data.csv', csv, function(err) {
            if (err) throw err;
            console.log('DONE');
          });
        });
      });
    });
}

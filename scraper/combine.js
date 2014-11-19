var fs = require('fs');
var json2csv = require('json2csv');

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

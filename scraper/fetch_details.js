var Promise = require('bluebird');
var csv = Promise.promisifyAll(require('fast-csv'));
var fs = Promise.promisifyAll(require('fs'));
var jsdom = Promise.promisifyAll(require('jsdom'));
var json2csv = require('json2csv');
var sys = Promise.promisifyAll(require('sys'));

var DEST_CODE = process.argv[2];
console.log(DEST_CODE);
var LISTINGS_FILE = 'data/listings.csv';
var ID_FILE = 'ids/' + DEST_CODE + '.txt';
var ERROR_FILE = DEST_CODE + '_errors.txt';
var ERRORS = fs.readFileSync(ERROR_FILE).toString().split('\n');

fetchExtraDetails(ID_FILE);

function getDestCode(url) {
  var dest = '';
  if (url.length) {
    var m = url.match(/CodeDest=(.+)&NoMLS=/);
    if (m.length > 1) {
      dest = m[1];
    }
  }
  return dest;
}

function extractIds() {
  var H = {}
  var stream = fs.createReadStream(LISTINGS_FILE);
  var csvStream = csv({ headers: true })
      .on("data", function(data) {
        var dest = getDestCode(data.PasserelleUrl);
        if (dest.length > 0) {
          H[dest] = H[dest] || [];
          H[dest].push('MT' + data.MlsNumber);
        }
      })
      .on("end", function() {
        for (var dest in H) {
          var fileName = 'ids/' + dest + '.txt';
          H[dest].forEach(function (id) {
            fs.appendFileSync(fileName, id + '\n');
          });
        }
      });
  stream.pipe(csvStream);
}

function fetchExtraDetails(fileName) {
  var mls_ids = fs.readFileSync(fileName).toString().split('\n');
  mls_ids.pop();
  //mls_ids.reverse();
  return Promise.all(mls_ids.map(function (x) { return fetchRemax(x); }))
                .then(function () {
                  console.log('DONE');
                });
  /*
  var sequencer = Promise.resolve();
  mls_ids.forEach(function (id, i) {
    sequencer = sequencer.then(function () {
      return fetchRemax(id);
    });
  });
  sequencer.then(function () {
    console.log('DONE');
  });
  */
}

function fetchRemax(id) {
  //var id = 'MT28773784';
  if (ERRORS.indexOf(id) > -1 || fs.existsSync('extra_data/' + id + '.json')) {
    console.log(id);
    return Promise.resolve();
  }
  var url = 'http://www.remax-quebec.com/en/MLSRedirect.rmx?sia=' + id;
  return jsdom.envAsync(url, ["http://code.jquery.com/jquery.js"])
    .then(function (window) {
      var $ = window.$;
      var fields = {
        'Year built :': 'YearBuilt',
        'Garage :': 'Garage',
        'Floor Covering :': 'FloorCovering',
        'Area :': 'Area',
        'Zoning :': 'Zoning',
        'Parking :': 'Parking',
        'Proximity :': 'Proximity',
        'Municipal (': 'MunicipalTax',
        'School (': 'SchoolTax',
        'Municipal assessment': 'MunicipalAssessment',
        'Insurance :': 'Insurance',
        'Heating Energy :': 'HeatingEnergy',
        'Water Supply :': 'WaterSupply',
        'Sewage System :': 'SewageSystem',
        'View :': 'View',
        'Topography :': 'Topography',
        'Living area :': 'LivingArea',
        'Pool :': 'Pool',
      };
      var data = { MlsNumber: id.substring(2) };
      for (var field in fields) {
        var key = fields[field];
        var res = $('td:contains("' + field + '")');
        if (res && res.next() && res.next().text().trim().length) {
          data[key] = res.next().text().trim();
        }
      }
      console.log(id);
      if (Object.keys(data).length > 1) {
        fs.writeFileSync('extra_data/' + id + '.json', JSON.stringify(data));
      }
  })
  .error(function () {
    fs.appendFileSync(ERROR_FILE, id+'\n');
    return Promise.resolve();
  });
}

/*
[ [ 'YANNDERY', 1 ],
  [ 'WEBLOFT', 1 ],
  [ 'PUBLIIMMO', 1 ],
  [ 'EXPANSION', 1 ],
  [ 'SALLYPHAN', 1 ],
  [ 'JUSTEPRIX', 1 ],
  [ 'INTERCOURTIERS', 1 ],
  [ 'ROYALCO', 1 ],
  [ 'EXPANSIONIMMO', 1 ],
  [ 'GROUPEGESTHEQUE', 1 ],
  [ 'IMMOWEB', 1 ],
  [ 'CREOMAX', 1 ],
  [ 'LOGIQC', 1 ],
  [ 'DOMICILIA', 1 ],
  [ 'LOUISEBELLEGARDE', 1 ],
  [ 'RDH', 1 ],
  [ 'TANDEM', 2 ],
  [ 'CENTOR', 2 ],
  [ 'JLL', 2 ],
  [ 'INFOEXPERT', 2 ],
  [ 'IMMOSTONE', 3 ],
  [ 'EVEREST', 3 ],
  [ 'IMMOBILIERGC', 3 ],
  [ 'QUBE', 4 ],
  [ 'JUTRAS', 5 ],
  [ 'ECORESEAU', 5 ],
  [ 'INTELLIPRO', 6 ],
  [ 'ZONE', 6 ],
  [ 'COMPAS', 6 ],
  [ 'FELIXJASMIN', 7 ],
  [ 'GOIMMO', 7 ],
  [ 'IMMEUBLESHOMEPRO', 7 ],
  [ 'HIMALAYA', 7 ],
  [ 'IEPANHOREA', 8 ],
  [ 'COURTIERASSOCIES', 8 ],
  [ 'LIMEMEDIA', 9 ],
  [ 'PLUSMAX', 9 ],
  [ 'IMMCO', 9 ],
  [ 'EXCEL', 9 ],
  [ 'IMMOBILIERBARON', 9 ],
  [ 'ORDINATEURLAVAL', 10 ],
  [ 'ANJES', 10 ],
  [ 'MCGUIGAN', 10 ],
  [ 'EXCLUSIVEMEDIA', 12 ],
  [ 'SPHERIKA', 12 ],
  [ 'ENNA', 13 ],
  [ 'DEAKIN', 13 ],
  [ 'CAROMTEX', 14 ],
  [ 'SERGEGABRIEL', 14 ],
  [ 'CHARISMA', 16 ],
  [ 'RESIDIA', 16 ],
  [ 'ALOUERMONTREAL', 17 ],
  [ 'SONIALAVOIE', 17 ],
  [ 'MAMAISON', 17 ],
  [ 'WHAMDI', 18 ],
  [ 'VORTEX', 23 ],
  [ 'NINAMILLER', 23 ],
  [ 'PREVISITE', 25 ],
  [ 'VANTAGEREALTYGROUP', 27 ],
  [ 'EBEAUDET', 27 ],
  [ 'REALTA', 28 ],
  [ 'ABBEYANDOLIVIER', 29 ],
  [ 'PLATEAU', 33 ],
  [ 'KONTACT', 36 ],
  [ 'WEBHDT', 44 ],
  [ 'MINIMAL', 54 ],
  [ 'LKAUFMAN', 58 ],
  [ 'DYNASIMPLE', 64 ],
  [ 'JMONTANARO', 65 ],
  [ 'ADRESZ', 66 ],
  [ 'MURAMUR', 67 ],
  [ 'KWURBAIN', 68 ],
  [ 'PERLMKTG', 81 ],
  [ 'CAMELEON', 88 ],
  [ 'VENDIRECT', 108 ],
  [ 'ID3', 130 ],
  [ 'LONDONOGROUP', 146 ],
  [ 'PROFUSION', 148 ],
  [ 'MCGILLIMMOBILIER', 155 ],
  [ 'MACLE', 176 ],
  [ 'KRYZALID', 212 ],
  [ 'RCIIQ', 304 ],
  [ 'EXPERT', 316 ],
  [ 'INDUSTRY44', 372 ],
  [ 'PROPRIO', 397 ],
  [ 'MELLOR', 403 ],
  [ 'VIACAPITALE', 508 ],
  [ 'ROYALLEPAGE', 514 ],
  [ 'C21', 708 ],
  [ 'SUTTON', 2258 ],
  [ 'EGPTECH', 2394 ],
  [ 'REMAX', 3408 ] ]
*/
function dumpStats() {
  var H = {};

  for (var i = 1; i <= 3668; ++i) {
    var data = JSON.parse(fs.readFileSync('responses/' + i + '.json').toString());
    data.d.Result.forEach(function (x) {
      var url = x.PasserelleUrl;
      if (url.length) {
        var m = url.match(/CodeDest=(.+)&NoMLS=/);
        if (m.length > 1) {
          var dest = m[1];
          H[dest] = H[dest] || 0;
          H[dest] += 1;
        }
      }
    });
  }

  var sortable = [];
  for (var dest in H) sortable.push([dest, H[dest]]);
  sortable.sort(function(a, b) {return a[1] - b[1]});
  console.log(sortable);
}

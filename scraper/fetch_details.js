var Promise = require('bluebird');
var csv = Promise.promisifyAll(require('fast-csv'));
var fs = Promise.promisifyAll(require('fs'));
var jsdom = Promise.promisifyAll(require('jsdom'));
var json2csv = require('json2csv');
var sys = Promise.promisifyAll(require('sys'));
var request = Promise.promisifyAll(require('request'));

var ENABLE_PARALLEL_FETCH = false;
var DEST_CODE = process.argv[2];
console.log(DEST_CODE);
var LISTINGS_FILE = 'data/listings.csv';
var ID_FILE = 'ids/' + DEST_CODE + '.txt';
var ERROR_FILE = DEST_CODE + '_errors.txt';
var ERRORS = fs.readFileSync(ERROR_FILE).toString().split('\n');

//fetchSUTTON('MT25144213');
fetchExtraDetails(ID_FILE);

function strip(s, sep) {
  return s.split('\t').map(function (x) { return x.trim(); }).filter(function (x) { return x.length > 0; }).join(', ');
}

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
  if (ENABLE_PARALLEL_FETCH) {
    return Promise.all(mls_ids.map(function (x) { return fetch(x); }))
                  .then(function () {
                    console.log('DONE');
                  });
  } else {
    var sequencer = Promise.resolve();
    mls_ids.forEach(function (id, i) {
      sequencer = sequencer.then(function () {
        return fetch(id);
      });
    });
    sequencer.then(function () {
      console.log('DONE');
    });
  }
}

function fetch(id) {
  var code = DEST_CODE === 'EGPTECH' ? 'ROYALLEPAGE' : DEST_CODE;
  if (ERRORS.indexOf(id) > -1 || fs.existsSync('extra_data/' + DEST_CODE + '/' + id + '.json')) {
    console.log(id);
    return Promise.resolve();
  }
  var fn = eval('fetch' + code);
  if (fn) {
    return fn(id);
  } else {
    throw 'Unknown dest code: ' + DEST_CODE;
  }
}

function fetchMELLOR(id) {
  //var id = 'MT28474580';
  var url = 'http://www.mellorgroup.ca/PropertyDetails?MLS=' + id.substring(2);
  return request.getAsync({
    url: url,
    headers: {
      'Accept': 'application/json, text/javascript, */*; q=0.01',
      'Content-Type': 'application/json; charset=UTF-8',
      'Host': 'www.mellorgroup.ca',
      'Origin': 'http://www.mellorgroup.ca',
      'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36',
    }
  }).get(1).then(function (html) {
    return jsdom.envAsync(html, ["http://code.jquery.com/jquery.js"])
      .then(function (window) {
        var $ = window.$;
        var data = {};
        [].forEach.call($('script'), function (x) {
          if (x.innerHTML.indexOf('__Noesis_Resources__') > -1) {
            eval(x.innerHTML);
            data = JSON.parse(__Noesis_Resources__.Settings.SerializedInscription);
          }
        });
        data.MlsNumber = id.substring(2);

        console.log(id);
        if (Object.keys(data).length > 1) {
          fs.writeFileSync('extra_data/' + DEST_CODE + '/' + id + '.json', JSON.stringify(data));
        }

        window.close();
    })
    .error(function () {
      fs.appendFileSync(ERROR_FILE, id+'\n');
      return Promise.resolve();
    });
  });
}

function fetchVIACAPITALE(id) {
  //var id = 'MT18238598';
  var url = 'http://redirect.viacapitale.com/no/' + id + '/?en=';
  return jsdom.envAsync(url, ["http://code.jquery.com/jquery.js"])
    .then(function (window) {
      var $ = window.$;
      // missing: FloorCovering, Insurance, View
      var fields = {
        'Land area :': 'Area',
        'Building:': 'BuildingAssessment',
        'Building :': 'BuildingAssessment',
        'Year of construction': 'YearBuilt',
        'Building area :': 'LivingArea',
        'Living area :': 'LivingArea',
        'Garage :': 'Garage',
        'Heating energy :': 'HeatingEnergy',
        'Heating system :': 'HeatingSystem',
        'Land:': 'LotAssessment',
        'Land :': 'LotAssessment',
        'Depth of land :': 'LotDepth',
        'Frontage land :': 'LotFrontage',
        'Condo fees': 'CondoFees', //
        'Municipal Taxes': 'MunicipalTax',
        'Parking :': 'Parking',
        'Swimming pool:': 'Pool',
        'Proximity :': 'Proximity',
        'School taxes': 'SchoolTax',
        'Water supply :': 'WaterSupply',
        'Sewage system :': 'SewageSystem',
        'Topography :': 'Topography',
        'Zoning :': 'Zoning'
      };
      var data = { MlsNumber: id.substring(2) };
      for (var field in fields) {
        var key = fields[field];
        var res = $('div:contains("' + field + '")');
        if (['YearBuilt'].indexOf(key) > -1) {
          if (res && res.length > 0 && res.last()) {
            var value = res.last().next().text().replace(/\n/g, '').replace(/ +(?= )/g,'').trim();
            value = value.substring(value.indexOf(':')+1).trim();
            if (value.length > 0) {
              data[key] = value;
            }
          }
        } else if (['BuildingAssessment', 'LotAssessment', 'MunicipalTax', 'SchoolTax'].indexOf(key) > -1) {
          if (res && res.length > 0 && res.last()) {
            var value = res.last().children().last().text().trim();
            if (field.indexOf(':') > -1 || ['MunicipalTax', 'SchoolTax'].indexOf(key) > -1) {
              value = res.last().text().trim();
            }
            value = value.substring(value.indexOf(':')+1).trim();
            if (value.length > 0) {
              data[key] = value;
            }
          }
        } else {
          if (res && res.length > 0 && res.last()) {
            var value = res.last().parent().text().replace(/\n/g, '').replace(/ +(?= )/g,'').trim();
            value = value.substring(value.indexOf(':')+1).trim();
            if (value.length > 0) {
              data[key] = value;
            }
          }
        }
      }
      console.log(id);
      if (Object.keys(data).length > 1) {
        fs.writeFileSync('extra_data/' + DEST_CODE + '/' + id + '.json', JSON.stringify(data));
      }

      window.close();
  })
  .error(function () {
    fs.appendFileSync(ERROR_FILE, id+'\n');
    return Promise.resolve();
  });
}

function fetchROYALLEPAGE(id) {
  //var id = 'MT13434401';
  var url = 'http://www-p.royallepage.ca/search/MLS-P/' + id;
  return jsdom.envAsync(url, ["http://code.jquery.com/jquery.js"])
    .then(function (window) {
      var $ = window.$;
      // missing: FloorCovering, Area, Insurance, Topography
      var fields = {
        'Building Assessment:': 'BuildingAssessment',
        'Built in': 'YearBuilt',
        'Floor Space (approx):': 'LivingArea',
        'Garage:': 'Garage',
        'Heating Energy:': 'HeatingEnergy',
        'Heating System:': 'HeatingSystem',
        'Lot Assessment:': 'LotAssessment',
        'Lot Depth:': 'LotDepth',
        'Lot Frontage:': 'LotFrontage',
        'Lot Size:': 'Area',
        'Maintenance Fees': 'CondoFees',
        'Municipal Tax:': 'MunicipalTax',
        'Parking:': 'Parking',
        'Pool:': 'Pool',
        'Proximity:': 'Proximity',
        'School Tax:': 'SchoolTax',
        'Water Supply:': 'WaterSupply',
        'View:': 'View',
        'Sewage System:': 'SewageSystem',
        'Zoning:': 'Zoning'
      };
      var data = { MlsNumber: id.substring(2) };
      for (var field in fields) {
        var key = fields[field];
        var res = $('li:contains("' + field + '")');
        if (res && res.length > 0 && res.next()) {
          var value = res.next().text().trim();
          if (value.length > 0) {
            data[key] = value;
          }
        }
      }
      console.log(id);
      if (Object.keys(data).length > 1) {
        fs.writeFileSync('extra_data/' + DEST_CODE + '/' + id + '.json', JSON.stringify(data));
      }

      window.close();
  })
  .error(function () {
    fs.appendFileSync(ERROR_FILE, id+'\n');
    return Promise.resolve();
  });
}

function fetchC21(id) {
  //var id = 'MT13434401';
  var url = 'http://www.century21.ca/' + id;
  return jsdom.envAsync(url, ["http://code.jquery.com/jquery.js"])
    .then(function (window) {
      var $ = window.$;
      // missing: FloorCovering, Area, Insurance, Topography
      var fields = {
        'Living Area:': 'LivingArea',
        'Lot Area:': 'Area',
        'Year Built:': 'YearBuilt',
        'Municipal:': 'MunicipalTax',
        'School:': 'SchoolTax',
        'Lot:': 'LotAssessment',
        'Building:': 'BuildingAssessment',
        'Condo Fees:': 'CondoFees',

        'Garage': 'Garage',
        'Heating Energy': 'HeatingEnergy',
        'Heating System': 'HeatingSystem',
        'Parking': 'Parking',
        'Pool': 'Pool',
        'Proximity': 'Proximity', // TODO: Handle multiple rows
        'Sewage System': 'SewageSystem',
        'View': 'View',
        'Water Supply': 'WaterSupply',
        'Zoning': 'Zoning'
      };
      var data = { MlsNumber: id.substring(2) };
      for (var field in fields) {
        var key = fields[field];
        if (['Zoning', 'Water Supply', 'View', 'SewageSystem', 'Proximity', 'Pool', 'Parking', 'HeatingSystem', 'HeatingEnergy', 'Garage'].indexOf(key) > -1) {
          var res = $('td:contains("' + field + '")');
          if (res && res.length > 0 && res.parent()) {
            var value = res.parent().next().text().split('\n').map(function (x) { return strip(x); }).filter(function (x) { return x.length > 0; }).join(', ');

            if (value.length > 0) {
              data[key] = value;
            }
          }
        } else {
          var res = $('td:contains("' + field + '")');
          if (res && res.next()) {
            var value = res.next().text().split('\n').map(function (x) { return strip(x); }).filter(function (x) { return x.length > 0; }).join(', ');
            if (value.length > 0) {
              data[key] = value;
            }
          }
        }
      }
      console.log(id);
      if (Object.keys(data).length > 1) {
        fs.writeFileSync('extra_data/' + DEST_CODE + '/' + id + '.json', JSON.stringify(data));
      }

      window.close();
  })
  .error(function () {
    fs.appendFileSync(ERROR_FILE, id+'\n');
    return Promise.resolve();
  });

}

function fetchSUTTON(id) {
  //var id = 'MT27529831';
  var url = 'http://www.suttonquebec.com/property/sutton-quebec-real-estate-details.html?no_inscription=' + id;
  return jsdom.envAsync(url, ["http://code.jquery.com/jquery.js"])
    .then(function (window) {
      var $ = window.$;
      // missing: FloorCovering, Area, Insurance, Topography, Pool
      var fields = {
        'year of constr.:': 'YearBuilt',
        'Parking': 'Parking',
        'Sewage system': 'SewageSystem',
        'Zoning': 'Zoning',
        'View': 'View',
        'Garage': 'Garage',
        'Proximity': 'Proximity',
        'Water supply': 'WaterSupply',
        'Bulding dim.': 'LivingArea',
        'Livable surface': 'LivingArea',
        'Lot surface': 'Area',
        'Municipal Taxes': 'MunicipalTax',
        'School taxes': 'SchoolTax',
        'Heating system': 'HeatingSystem',
        'Municipal evaluation': 'MunicipalAssessment',
      };
      var data = { MlsNumber: id.substring(2) };
      for (var field in fields) {
        var key = fields[field];
        if (key === 'MunicipalAssessment') {
          var res = $('div:contains("' + field + '")');
          if (res.length > 0 && res.last()) {
            var value = res.last().text().split('\n')[2].trim().replace(/ \([0-9]+\)/, '');
            if (value.length > 0) {
              data[key] = value;
            }
          }
        } else {
          var res = $('td:contains("' + field + '")');
          if (res && res.next()) {
            var value = res.next().text().split('\n').map(function (x) { return x.trim(); }).filter(function (x) { return x.length > 0; }).join(', ');
            if (value.length > 0) {
              data[key] = value;
            }
          }
        }
      }
      console.log(id);
      if (Object.keys(data).length > 1) {
        fs.writeFileSync('extra_data/' + DEST_CODE + '/' + id + '.json', JSON.stringify(data));
      }
      
      window.close();
  })
  .error(function () {
    fs.appendFileSync(ERROR_FILE, id+'\n');
    return Promise.resolve();
  });
}

function fetchREMAX(id) {
  //var id = 'MT28773784';
  var url = 'http://www.remax-quebec.com/en/MLSRedirect.rmx?sia=' + id;
  return jsdom.envAsync(url, ["http://code.jquery.com/jquery.js"])
    .then(function (window) {
      var $ = window.$;
      // missing: HeatingSystem
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
        fs.writeFileSync('extra_data/' + DEST_CODE + '/' + id + '.json', JSON.stringify(data));
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

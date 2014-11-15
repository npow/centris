var fs = require('fs');

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

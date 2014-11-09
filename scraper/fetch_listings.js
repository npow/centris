var Promise = require('bluebird');
var fs = require('fs');
var request = Promise.promisifyAll(require('request'));

var url = 'http://www.centris.ca/Services/PropertyService.asmx/GetPropertyViews';

function delay(ms) {
  var deferred = Promise.pending();
  setTimeout(function(){
    deferred.fulfill();
  }, ms);
  return deferred.promise;
}

function fetchPage(pageIndex) {
  return request
    .postAsync({
      url: url,
      body: JSON.stringify({ pageIndex: pageIndex, track: false }),
      headers: {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Content-Type': 'application/json; charset=UTF-8',
        'Host': 'www.centris.ca',
        'Origin': 'http://www.centris.ca',
        'Referer': 'http://www.centris.ca/en/property~for-sale~montreal-island',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36',
        'X-CENTRIS-UC': 3,
        'X-CENTRIS-UCK': '280419d8-fe78-47e5-a5d8-5ed120f1ee24',
        'X-Requested-With': 'XMLHttpRequest',
        'Cookie': 'ASP.NET_SessionId=34m5yxrwbaymlc0dfiribirl; Centris.PropertySearchFavorites=; Centris.PropertySearchRemoved=; Centris.BrokerSearchFavorites=; Centris=Token=0a37e347-ec05-4b02-bd9e-be66c4fede5b&Lang=en&Reg=GSGS4621&PropertySearchView=List&PromotionVersion=0&PromotionPopupShownCount=2; Centris.AllowMobileRedirection=1415515707000%7Cfull'
      }
    })
    .get(1)
    .then(function (response) {
      console.log(pageIndex);
      fs.writeFile('responses/' + pageIndex + '.json', response);
    })
    .error(function (err) {
      console.log('ERROR: ' + pageIndex);
    });
}

var sequencer = Promise.resolve();
for (var i = 1; i <= 3668; ++i) {
  sequencer = sequencer.then(function () {
    return delay(10);
  }).then((function (i) {
    return function () {
      return fetchPage(i);
    };
  })(i));
}
sequencer.then(function () {
  console.log('DONE');
});

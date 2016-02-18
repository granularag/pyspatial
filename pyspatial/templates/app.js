//Map
L.Icon.Default.imagePath = "{{ static_img }}";

PL.colors.initializeColorBrewer();

var DATA = JSON.parse($("#data").html());

var app = angular.module('main', ['dsChart', 'templates-dist']);

function getPalette(cp) {
  var p = DATA.choropleths[cp];
  if (_.has(p, "palette")) {
    return PL.colors.colorBrewer[p.palette][p.levels];
  } else {
    return p;
  }
}

app.controller('controller', ['$scope', '$timeout', '$http', '$q', 'tileService', function($scope, $timeout, $http, $q, tileService) {

  var baseMaps = {true: {"name": "cartodb-dark", "text": "Light Map"},
                  false: {"name":"cartodb", "text":"Dark Map"}};

  $scope.ds = DS.fromJSON(DATA.dataset);
  $scope.center = DATA.view;
  $scope.map = new PL.Map("lmap", $scope.center.lat, $scope.center.lng,
                          $scope.ds, $scope.center.zoom);


  $scope.map.map.on('baselayerchange', function (eventLayer) {
    var cp = eventLayer.name;
    var palette = getPalette(cp);
    $scope.map.choropleth(DATA.base_layer, cp, palette, DATA.info_cols);
  });

  $scope.tiles = null;
  $scope.basemap = true;

  $scope.toggleBaseMap = function(basemap) {
    if ($scope.tiles !== null ) {
      $scope.map.map.removeLayer($scope.tiles);
      $scope.tiles = null;
    }
    $scope.basemap = !basemap;
    var bm = baseMaps[$scope.basemap];
    $scope.tiles = tileService($scope.map, bm.name);
    angular.element("#tile-toggle").html(bm.text);
  };

  $scope.toggleBaseMap($scope.basemap);

  for (var cp in DATA.choropleths) {
    var palette = getPalette(cp);
    $scope.map.choropleth(DATA.base_layer, cp, palette, DATA.info_cols);
  }
  console.log(DATA.overlays);
  for (var o in DATA.overlays) {
    $scope.map.addOverlay(o, DATA.overlays[o].kind, DATA.overlays[o].shape,
                          DATA.overlays[o].style,
                          DATA.overlays[o].text, false);
  }
}]);

// CẤU HÌNH MÔ HÌNH
var RESOLUTION = 30;
var NUM_TREES = 1000; // Optimized from PSO (iteration 12+, R²=0.7742, Fitness=0.4355)
var TRAIN_SPLIT = 0.7;
var BUFFER_SIZE = 15; // Buffer 15m cho điểm để lấy mẫu tốt hơn

var MIN_LEAF_POPULATION = 1;     // min_samples_leaf from PSO (optimal value)
var VARIABLES_PER_SPLIT = null;  // max_features='sqrt' → GEE default (sqrt of features ≈ 3.6)
var BAG_FRACTION = 0.5;          // Bootstrap=False in PSO, reduced bagging

var featureNames = [
  'lulc', 'Density_River', 'Density_Road', 'Distan2river', 'Distan2road_met',
  'aspect', 'curvature', 'dem', 'flowDir', 'slope', 'twi', 'NDVI', 'rainfall'
];


var interpolateNulls = function(image, bandNames, radius, iterations) {
  var interpolated = image;
  
  for (var i = 0; i < iterations; i++) {
    var newBands = bandNames.map(function(bandName) {
      var band = interpolated.select([bandName]);
      var validMask = band.mask();
      var filled = band.focal_mean({
        radius: radius,
        units: 'meters',
        kernelType: 'square'
      });
      var result = band.unmask(filled);
      return result.rename(bandName);
    });
    
    interpolated = ee.Image(newBands);
  }
  
  return interpolated;
};

//////////////////////////////////////////////////////////////
// FUNCTION: THỐNG KÊ NULL VALUES
//////////////////////////////////////////////////////////////

var countNullPixels = function(image, bandNames, region, scale, label) {
  print('\n--- ' + label + ' ---');
  
  var stats = ee.List(bandNames).map(function(bandName) {
    var band = image.select([bandName]);
    
    var totalPixels = band.unmask(-9999).reduceRegion({
      reducer: ee.Reducer.count(),
      geometry: region,
      scale: scale,
      maxPixels: 1e9
    });
    
    var validPixels = band.reduceRegion({
      reducer: ee.Reducer.count(),
      geometry: region,
      scale: scale,
      maxPixels: 1e9
    });
    
    var total = ee.Number(totalPixels.get(bandName));
    var valid = ee.Number(validPixels.get(bandName));
    var nullCount = total.subtract(valid);
    var nullPercent = nullCount.divide(total).multiply(100);
    
    return ee.Dictionary({
      'band': bandName,
      'total': total,
      'valid': valid,
      'null': nullCount,
      'null_percent': nullPercent
    });
  });
  
  print(stats);
};

//////////////////////////////////////////////////////////////
// BƯỚC 1: LOAD ASSETS
//////////////////////////////////////////////////////////////

print('\n========== STEP 1: LOADING ASSETS ==========');

if (typeof studyArea === 'undefined') {
  print('ERROR: studyArea not imported!');
}
if (typeof imageAsset === 'undefined') {
  print('ERROR: imageAsset not imported!');
}
if (typeof floodPoints === 'undefined') {
  print('ERROR: floodPoints not imported!');
}

var floodPoints = floodPoints.map(function(feature) {
  var lon = ee.Number(feature.get('lon'));
  var lat = ee.Number(feature.get('lat'));
  var floodValue = ee.Number(feature.get('flood'));
  return feature.set('flood', floodValue)
                .setGeometry(ee.Geometry.Point([lon, lat], 'EPSG:4326'));
});

print('✅ Assets loaded');
print('Total flood points:', floodPoints.size());
print('Flood points (1):', floodPoints.filter(ee.Filter.eq('flood', 1)).size());
print('Non-flood points (0):', floodPoints.filter(ee.Filter.eq('flood', 0)).size());

//////////////////////////////////////////////////////////////
// BƯỚC 2: EXTRACT FEATURES VỚI BUFFER
//////////////////////////////////////////////////////////////

print('\n========== STEP 2: EXTRACTING FEATURES ==========');

var features = imageAsset.select(featureNames).clip(studyArea);

var floodPointsBuffered = floodPoints.map(function(feature) {
  var buffered = feature.buffer(BUFFER_SIZE);
  return buffered.copyProperties(feature, ['flood', 'lat', 'lon']);
});

print('✅ Applied ' + BUFFER_SIZE + 'm buffer to points');

var trainingData = features.sampleRegions({
  collection: floodPointsBuffered,
  properties: ['flood', 'lat', 'lon'],
  scale: 10,
  tileScale: 1,
  geometries: false
});

print('Total points after sampling:', trainingData.size());

print('\n--- Checking null values in training samples ---');
featureNames.forEach(function(bandName) {
  var countNonNull = trainingData.filter(ee.Filter.notNull([bandName])).size();
  print('  ' + bandName + ':', countNonNull);
});

var requiredColumns = ['flood'].concat(featureNames);
trainingData = trainingData.filter(ee.Filter.notNull(requiredColumns));

print('✅ Valid training samples:', trainingData.size());

var withRandom = trainingData.randomColumn('random', 42);
var training = withRandom.filter(ee.Filter.lt('random', TRAIN_SPLIT));
var validation = withRandom.filter(ee.Filter.gte('random', TRAIN_SPLIT));

print('Training samples:', training.size());
print('Validation samples:', validation.size());

//////////////////////////////////////////////////////////////
// BƯỚC 3: TRAIN MODEL
//////////////////////////////////////////////////////////////

print('\n========== STEP 3: TRAINING MODEL ==========');

// Random Forest with PSO optimized parameters (best from iteration 12+)
var rfRegressor = ee.Classifier.smileRandomForest({
  numberOfTrees: NUM_TREES,              // 1000 trees (PSO optimal)
  variablesPerSplit: VARIABLES_PER_SPLIT, // null = sqrt(features) ≈ 3.6 features
  minLeafPopulation: MIN_LEAF_POPULATION, // 1 (optimal value)
  bagFraction: BAG_FRACTION,              // 0.5 (bootstrap disabled in PSO)
  seed: 42
}).setOutputMode('REGRESSION')
  .train({
    features: training,
    classProperty: 'flood',
    inputProperties: featureNames
  });

print('✅ Model trained with PSO optimized parameters');
print('  PSO Best: R²=0.7742, Fitness=0.4355, MAE=0.1098, RMSE=0.2289');
print('  Parameters: 1000 trees, sqrt features/split, min_leaf=1, bag_fraction=0.5');
print('  Note: GEE does not support max_depth(50), max_leaf_nodes(1000), min_samples_split(20)');
print('  PSO outperforms Random Search (R²=0.7325) by +5.7%');


//////////////////////////////////////////////////////////////
// BƯỚC 4: VALIDATION METRICS
//////////////////////////////////////////////////////////////

print('\n========== STEP 4: VALIDATION METRICS ==========');

var validationPredicted = validation.classify(rfRegressor, 'predicted');

var validationWithErrors = validationPredicted.map(function(f) {
  var observed = ee.Number(f.get('flood'));
  var predicted = ee.Number(f.get('predicted'));
  var error = observed.subtract(predicted);
  var squaredError = error.pow(2);
  var absError = error.abs();
  return f.set({
    'error': error,
    'squared_error': squaredError,
    'abs_error': absError
  });
});

var rmse = ee.Number(validationWithErrors.aggregate_mean('squared_error')).sqrt();
var mae = validationWithErrors.aggregate_mean('abs_error');

var meanObserved = validationPredicted.aggregate_mean('flood');

var validationWithSST = validationPredicted.map(function(f) {
  var observed = ee.Number(f.get('flood'));
  var predicted = ee.Number(f.get('predicted'));
  var residual = observed.subtract(predicted);
  var totalDeviation = observed.subtract(meanObserved);
  return f.set({
    'ss_res': residual.pow(2),
    'ss_tot': totalDeviation.pow(2)
  });
});

var ss_res = validationWithSST.aggregate_sum('ss_res');
var ss_tot = validationWithSST.aggregate_sum('ss_tot');
var r2 = ee.Number(1).subtract(ee.Number(ss_res).divide(ss_tot));

var validationList = validationPredicted.reduceColumns({
  reducer: ee.Reducer.pearsonsCorrelation(),
  selectors: ['flood', 'predicted']
});

print('--- Regression Metrics ---');
print('R² (Coefficient of Determination):', r2);
print('Pearson Correlation:', validationList.get('correlation'));
print('RMSE:', rmse);
print('MAE:', mae);
print('Mean Observed Value:', meanObserved);

var importance = rfRegressor.explain();
print('\n--- Feature Importance ---');
print(importance);

//////////////////////////////////////////////////////////////
// BƯỚC 5: PREDICTION
//////////////////////////////////////////////////////////////

print('\n========== STEP 5: MAKING PREDICTIONS ==========');

var floodProbability = features.classify(rfRegressor).rename('flood_probability');
floodProbability = floodProbability.clamp(0, 1);

//////////////////////////////////////////////////////////////
// BƯỚC 5.5: KIỂM TRA VÀ NỘI SUY NULL TRONG PREDICTION
//////////////////////////////////////////////////////////////

print('\n========== STEP 5.5: CHECK & INTERPOLATE NULLS IN PREDICTION ==========');

countNullPixels(floodProbability, ['flood_probability'], studyArea, RESOLUTION * 2,
                'NULL COUNT BEFORE INTERPOLATION (Prediction)');

print('\n🔄 Interpolating null values in prediction...');
print('  Radius: 90m (3 pixels) | Iterations: 3');

var floodProbabilityInterpolated = interpolateNulls(
  floodProbability,
  ['flood_probability'],
  90,
  3
);

floodProbabilityInterpolated = floodProbabilityInterpolated.clamp(0, 1);

countNullPixels(floodProbabilityInterpolated, ['flood_probability'], studyArea, RESOLUTION * 2,
                'NULL COUNT AFTER INTERPOLATION (Prediction)');

print('✅ Prediction interpolation completed');

//////////////////////////////////////////////////////////////
// TÍNH RISK LEVELS (FIXED ALGORITHM)
//////////////////////////////////////////////////////////////

print('\n========== CALCULATING RISK LEVELS ==========');

// THUẬT TOÁN MỚI: Dùng expression để phân loại rõ ràng
var riskLevels = floodProbabilityInterpolated.expression(
  '(b1 < 0.25) ? 1 : ' +  // Low Risk
  '(b1 < 0.50) ? 2 : ' +  // Moderate Risk
  '(b1 < 0.75) ? 3 : 4',  // High Risk : Very High Risk
  {
    'b1': floodProbabilityInterpolated.select('flood_probability')
  }
).rename('risk_level').toInt();

// Reproject outputs
floodProbabilityInterpolated = floodProbabilityInterpolated.reproject({
  crs: 'EPSG:4326',
  scale: RESOLUTION
}).clip(studyArea);

riskLevels = riskLevels.reproject({
  crs: 'EPSG:4326',
  scale: RESOLUTION
}).clip(studyArea);

print('✅ Risk levels calculated');

// Kiểm tra phân bố risk levels
var riskStats = riskLevels.reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(),
  geometry: studyArea,
  scale: RESOLUTION * 2,
  maxPixels: 1e9
});
print('\n--- Risk Level Distribution ---');
print('Histogram:', riskStats.get('risk_level'));

// Kiểm tra range probability
var probStats = floodProbabilityInterpolated.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: studyArea,
  scale: RESOLUTION * 10,
  maxPixels: 1e9
});
print('\nProbability range - Min:', probStats.get('flood_probability_min'), 
      '| Max:', probStats.get('flood_probability_max'));

//////////////////////////////////////////////////////////////
// BƯỚC 6: VISUALIZATION
//////////////////////////////////////////////////////////////

print('\n========== STEP 6: DISPLAYING RESULTS ==========');

Map.centerObject(studyArea, 10);

// Study area
Map.addLayer(studyArea, {color: '0000FF'}, 'Study Area', true, 0.3);

// Training points
var floodPts = floodPoints.filter(ee.Filter.eq('flood', 1));
var nonFloodPts = floodPoints.filter(ee.Filter.eq('flood', 0));
Map.addLayer(floodPts, {color: 'FF0000'}, 'Flood Points (1)', true);
Map.addLayer(nonFloodPts, {color: '00FF00'}, 'Non-Flood Points (0)', true);

// Flood probability (continuous)
var probPalette = {
  min: 0,
  max: 1,
  palette: ['#00FF00', '#FFFF00', '#FF9900', '#FF0000']
};
Map.addLayer(floodProbabilityInterpolated, probPalette, '🎯 Flood Probability', true);

// Risk levels (categorical)
var riskPalette = {
  min: 1,
  max: 4,
  palette: ['#00FF00', '#FFFF00', '#FF9900', '#FF0000']
};
Map.addLayer(riskLevels, riskPalette, '📊 Risk Levels (1-4)', true);

// Legend
var legend = ui.Panel({
  style: {position: 'bottom-left', padding: '8px 15px'}
});

var legendTitle = ui.Label({
  value: 'Flood Risk Levels',
  style: {fontWeight: 'bold', fontSize: '16px', margin: '0 0 4px 0'}
});
legend.add(legendTitle);

var makeRow = function(color, name) {
  var colorBox = ui.Label({
    style: {backgroundColor: color, padding: '8px', margin: '0 0 4px 0'}
  });
  var description = ui.Label({
    value: name,
    style: {margin: '0 0 4px 6px'}
  });
  return ui.Panel({
    widgets: [colorBox, description],
    layout: ui.Panel.Layout.Flow('horizontal')
  });
};

legend.add(makeRow('#00FF00', 'Level 1 (0.00-0.25): Low Risk'));
legend.add(makeRow('#FFFF00', 'Level 2 (0.25-0.50): Moderate Risk'));
legend.add(makeRow('#FF9900', 'Level 3 (0.50-0.75): High Risk'));
legend.add(makeRow('#FF0000', 'Level 4 (0.75-1.00): Very High Risk'));

Map.add(legend);

print('✅ Map layers added');

//////////////////////////////////////////////////////////////
// BƯỚC 7: EXPORT
//////////////////////////////////////////////////////////////

print('\n========== STEP 7: EXPORTING RESULTS ==========');

// Export probability map
Export.image.toDrive({
  image: floodProbabilityInterpolated,
  description: 'flood_probability_RF',
  scale: RESOLUTION,
  region: studyArea,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  crs: 'EPSG:4326'
});

// Export risk levels (với metadata đầy đủ)
Export.image.toDrive({
  image: riskLevels.toInt(),
  description: 'flood_risk_levels_RF',
  scale: RESOLUTION,
  region: studyArea,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  crs: 'EPSG:4326'
});

// Export validation results
Export.table.toDrive({
  collection: validationWithErrors.select(['lat', 'lon', 'flood', 'predicted', 'error', 'squared_error', 'abs_error']),
  description: 'validation_results_RF',
  fileFormat: 'CSV'
});

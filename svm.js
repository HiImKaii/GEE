// CẤU HÌNH MÔ HÌNH SVR - IMPROVED VERSION
var RESOLUTION = 30;
var TRAIN_SPLIT = 0.7;
var BUFFER_SIZE = 15;

// SVR Parameters - OPTIMIZED FOR HIGH ACCURACY (Target R²: 0.9)
var SVM_KERNEL = 'RBF';       // RBF kernel - best for complex patterns
var SVM_C = 150.0;            // High cost - allow more flexibility to fit data
var SVM_GAMMA = 0.15;         // Higher gamma - capture more complex patterns
var SVM_EPSILON = 0.01;       // Lower epsilon - stricter fitting

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
// FUNCTION: STANDARDIZATION (Z-SCORE) - Better than Min-Max for SVR
//////////////////////////////////////////////////////////////

var standardizeFeatures = function(data, featureNames) {
  print('\n🔧 Standardizing features with Z-Score (mean=0, std=1)...');
  
  // Tính mean và std cho từng feature
  var stats = featureNames.map(function(feature) {
    var values = data.aggregate_array(feature);
    var mean = values.reduce(ee.Reducer.mean());
    var stdDev = values.reduce(ee.Reducer.stdDev());
    
    return ee.Dictionary({
      'feature': feature,
      'mean': mean,
      'std': stdDev
    });
  });
  
  var statsDict = ee.Dictionary.fromLists(
    ee.List(stats).map(function(d) { return ee.Dictionary(d).get('feature'); }),
    stats
  );
  
  // Standardize: (x - mean) / std
  var standardized = data.map(function(f) {
    var props = {};
    
    featureNames.forEach(function(feature) {
      var stat = ee.Dictionary(statsDict.get(feature));
      var mean = ee.Number(stat.get('mean'));
      var std = ee.Number(stat.get('std'));
      
      var value = ee.Number(f.get(feature));
      
      // Tránh chia cho 0
      var standardizedValue = ee.Algorithms.If(
        std.eq(0),
        0,
        value.subtract(mean).divide(std)
      );
      
      props[feature + '_std'] = standardizedValue;
    });
    
    return f.set(props);
  });
  
  print('✅ Standardization completed for', featureNames.length, 'features');
  
  return ee.Dictionary({
    'data': standardized,
    'stats': statsDict
  });
};



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

var requiredColumns = ['flood'].concat(featureNames);
trainingData = trainingData.filter(ee.Filter.notNull(requiredColumns));

print('✅ Valid training samples:', trainingData.size());

//////////////////////////////////////////////////////////////
// BƯỚC 2.5: STANDARDIZATION ONLY
//////////////////////////////////////////////////////////////

print('\n========== STEP 2.5: STANDARDIZING FEATURES ==========');

// Standardize thay vì normalize
var stdResult = standardizeFeatures(trainingData, featureNames);
var trainingDataStd = ee.FeatureCollection(ee.Dictionary(stdResult).get('data'));
var stdStats = ee.Dictionary(ee.Dictionary(stdResult).get('stats'));

// Tạo danh sách các feature đã chuẩn hóa
var standardizedFeatureNames = featureNames.map(function(name) {
  return name + '_std';
});

print('Total features for training:', standardizedFeatureNames.length);
print('Feature names:', standardizedFeatureNames);

// Split train/validation với stratification
var flood1 = trainingDataStd.filter(ee.Filter.eq('flood', 1)).randomColumn('random', 42);
var flood0 = trainingDataStd.filter(ee.Filter.eq('flood', 0)).randomColumn('random', 42);

var training = flood1.filter(ee.Filter.lt('random', TRAIN_SPLIT))
                     .merge(flood0.filter(ee.Filter.lt('random', TRAIN_SPLIT)));
var validation = flood1.filter(ee.Filter.gte('random', TRAIN_SPLIT))
                       .merge(flood0.filter(ee.Filter.gte('random', TRAIN_SPLIT)));

print('Training samples:', training.size());
print('  - Flood (1):', training.filter(ee.Filter.eq('flood', 1)).size());
print('  - Non-flood (0):', training.filter(ee.Filter.eq('flood', 0)).size());
print('Validation samples:', validation.size());
print('  - Flood (1):', validation.filter(ee.Filter.eq('flood', 1)).size());
print('  - Non-flood (0):', validation.filter(ee.Filter.eq('flood', 0)).size());

//////////////////////////////////////////////////////////////
// BƯỚC 3: TRAINING ENSEMBLE SVR MODELS
//////////////////////////////////////////////////////////////

print('\n========== STEP 3: TRAINING ENSEMBLE SVR MODELS ==========');

// Model 1: RBF kernel với tham số cao để fit tốt hơn
var svr1 = ee.Classifier.libsvm({
  svmType: 'EPSILON_SVR',
  kernelType: SVM_KERNEL,
  gamma: SVM_GAMMA,
  cost: SVM_C
}).setOutputMode('REGRESSION')
  .train({
    features: training,
    classProperty: 'flood',
    inputProperties: standardizedFeatureNames
  });

// Model 2: RBF kernel với gamma cao hơn (phụ trợ)
var svr2 = ee.Classifier.libsvm({
  svmType: 'EPSILON_SVR',
  kernelType: 'RBF',
  gamma: 0.2,
  cost: 120.0
}).setOutputMode('REGRESSION')
  .train({
    features: training,
    classProperty: 'flood',
    inputProperties: standardizedFeatureNames
  });

// Model 3: Polynomial kernel cao cấp (phụ trợ)
var svr3 = ee.Classifier.libsvm({
  svmType: 'EPSILON_SVR',
  kernelType: 'POLY',
  gamma: 0.1,
  cost: 100.0,
  degree: 4
}).setOutputMode('REGRESSION')
  .train({
    features: training,
    classProperty: 'flood',
    inputProperties: standardizedFeatureNames
  });

print('✅ Trained 3 SVR models for ensemble');

//////////////////////////////////////////////////////////////
// BƯỚC 4: ENSEMBLE VALIDATION
//////////////////////////////////////////////////////////////

print('\n========== STEP 4: ENSEMBLE VALIDATION ==========');

var val1 = validation.classify(svr1, 'pred1');
var val2 = val1.classify(svr2, 'pred2');
var val3 = val2.classify(svr3, 'pred3');

// Ensemble: Trung bình có trọng số
var validationEnsemble = val3.map(function(f) {
  var p1 = ee.Number(f.get('pred1'));
  var p2 = ee.Number(f.get('pred2'));
  var p3 = ee.Number(f.get('pred3'));
  
  // Weighted average - tăng trọng số cho model chính
  var predicted = p1.multiply(0.5).add(p2.multiply(0.3)).add(p3.multiply(0.2));
  predicted = predicted.clamp(0, 1);
  
  var observed = ee.Number(f.get('flood'));
  var error = observed.subtract(predicted);
  var squaredError = error.pow(2);
  var absError = error.abs();
  
  return f.set({
    'predicted': predicted,
    'error': error,
    'squared_error': squaredError,
    'abs_error': absError
  });
});

var rmse = ee.Number(validationEnsemble.aggregate_mean('squared_error')).sqrt();
var mae = validationEnsemble.aggregate_mean('abs_error');

var meanObserved = validationEnsemble.aggregate_mean('flood');

var validationWithSST = validationEnsemble.map(function(f) {
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

var validationList = validationEnsemble.reduceColumns({
  reducer: ee.Reducer.pearsonsCorrelation(),
  selectors: ['flood', 'predicted']
});

print('--- ENSEMBLE Regression Metrics ---');
print('R² (Coefficient of Determination):', r2);
print('Pearson Correlation:', validationList.get('correlation'));
print('RMSE:', rmse);
print('MAE:', mae);
print('Mean Observed Value:', meanObserved);

//////////////////////////////////////////////////////////////
// BƯỚC 5: PREDICTION VỚI ENSEMBLE
//////////////////////////////////////////////////////////////

print('\n========== STEP 5: MAKING ENSEMBLE PREDICTIONS ==========');

// Standardize features cho prediction
var featuresStd = features;

featureNames.forEach(function(feature) {
  var stat = ee.Dictionary(stdStats.get(feature));
  var mean = ee.Number(stat.get('mean'));
  var std = ee.Number(stat.get('std'));
  
  var band = features.select([feature]);
  var standardizedBand = band.subtract(mean).divide(std.max(1e-10));
  
  featuresStd = featuresStd.addBands(
    standardizedBand.rename(feature + '_std'),
    null,
    true
  );
});

print('✅ Features standardized for prediction');

// Predict với 3 models
var pred1 = featuresStd.select(standardizedFeatureNames).classify(svr1).rename('pred1');
var pred2 = featuresStd.select(standardizedFeatureNames).classify(svr2).rename('pred2');
var pred3 = featuresStd.select(standardizedFeatureNames).classify(svr3).rename('pred3');

// Ensemble prediction
var floodProbability = pred1.multiply(0.4)
  .add(pred2.multiply(0.3))
  .add(pred3.multiply(0.3))
  .clamp(0, 1)
  .rename('flood_probability');

//////////////////////////////////////////////////////////////
// BƯỚC 5.5: NỘI SUY NULL
//////////////////////////////////////////////////////////////

print('\n========== STEP 5.5: INTERPOLATING NULLS ==========');

var floodProbabilityInterpolated = interpolateNulls(
  floodProbability,
  ['flood_probability'],
  90,
  3
);

floodProbabilityInterpolated = floodProbabilityInterpolated.clamp(0, 1);

print('✅ Prediction interpolation completed');

//////////////////////////////////////////////////////////////
// TÍNH RISK LEVELS
//////////////////////////////////////////////////////////////

print('\n========== CALCULATING RISK LEVELS ==========');

var riskLevels = floodProbabilityInterpolated.expression(
  '(b1 < 0.25) ? 1 : ' +
  '(b1 < 0.50) ? 2 : ' +
  '(b1 < 0.75) ? 3 : 4',
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

//////////////////////////////////////////////////////////////
// BƯỚC 6: VISUALIZATION
//////////////////////////////////////////////////////////////

print('\n========== STEP 6: DISPLAYING RESULTS ==========');

Map.centerObject(studyArea, 10);

Map.addLayer(studyArea, {color: '0000FF'}, 'Study Area', true, 0.3);

var floodPts = floodPoints.filter(ee.Filter.eq('flood', 1));
var nonFloodPts = floodPoints.filter(ee.Filter.eq('flood', 0));
Map.addLayer(floodPts, {color: 'FF0000'}, 'Flood Points (1)', true);
Map.addLayer(nonFloodPts, {color: '00FF00'}, 'Non-Flood Points (0)', true);

var probPalette = {
  min: 0,
  max: 1,
  palette: ['#00FF00', '#FFFF00', '#FF9900', '#FF0000']
};
Map.addLayer(floodProbabilityInterpolated, probPalette, '🎯 Flood Probability (Ensemble SVR)', true);

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
  value: 'Flood Risk Levels (Ensemble SVR)',
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

print('\n========== STEP 7: EXPORTING RESULTS ==========');

Export.image.toDrive({
  image: floodProbabilityInterpolated,
  description: 'flood_probability_Ensemble_SVR',
  scale: RESOLUTION,
  region: studyArea,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  crs: 'EPSG:4326'
});

Export.image.toDrive({
  image: riskLevels.toInt(),
  description: 'flood_risk_levels_Ensemble_SVR',
  scale: RESOLUTION,
  region: studyArea,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  crs: 'EPSG:4326'
});

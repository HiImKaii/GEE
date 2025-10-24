// ========================================================================
// FLOOD RISK PREDICTION MODEL USING RANDOM FOREST REGRESSION
// Bài toán: HỒI QUY - Dự đoán xác suất liên tục từ 0-1
// VỚI NỘI SUY NULL SAU KHI PREDICT
// ========================================================================

// CẤU HÌNH MÔ HÌNH
var RESOLUTION = 30;
var NUM_TREES = 200;
var TRAIN_SPLIT = 0.7;
var BUFFER_SIZE = 15; // Buffer 15m cho điểm để lấy mẫu tốt hơn

var featureNames = [
  'lulc', 'Density_River', 'Density_Road', 'Distan2river', 'Distan2road_met',
  'aspect', 'curvature', 'dem', 'flowDir', 'slope', 'twi', 'NDVI', 'rainfall'
];

print('========== FLOOD RISK PREDICTION - RANDOM FOREST REGRESSION ==========');
print('Resolution: ' + RESOLUTION + 'm | Trees: ' + NUM_TREES + ' | Buffer: ' + BUFFER_SIZE + 'm');

//////////////////////////////////////////////////////////////
// FUNCTION: NỘI SUY NULL VALUES
//////////////////////////////////////////////////////////////

var interpolateNulls = function(image, bandNames, radius, iterations) {
  /**
   * Nội suy các giá trị null trong image bằng focal mean
   * Chỉ thay thế null values, giữ nguyên các pixel có giá trị
   * 
   * @param {ee.Image} image - Image cần nội suy
   * @param {List} bandNames - Danh sách các band cần nội suy
   * @param {Number} radius - Bán kính kernel (meters)
   * @param {Number} iterations - Số lần lặp nội suy
   */
  
  var interpolated = image;
  
  for (var i = 0; i < iterations; i++) {
    var newBands = bandNames.map(function(bandName) {
      var band = interpolated.select([bandName]);
      
      // Tạo mask cho null values (0 = null, 1 = valid)
      var validMask = band.mask();
      
      // Tính focal mean của các pixel xung quanh
      var filled = band.focal_mean({
        radius: radius,
        units: 'meters',
        kernelType: 'square'
      });
      
      // Chỉ lấy giá trị nội suy cho null pixels
      // Giữ nguyên giá trị gốc cho valid pixels
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
  
  // Tạo một reducer tổng hợp cho tất cả bands
  var stats = ee.List(bandNames).map(function(bandName) {
    var band = image.select([bandName]);
    
    // Đếm total pixels (bao gồm cả null)
    var totalPixels = band.unmask(-9999).reduceRegion({
      reducer: ee.Reducer.count(),
      geometry: region,
      scale: scale,
      maxPixels: 1e9
    });
    
    // Đếm valid pixels (chỉ những pixel có mask = 1)
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
  
  // In kết quả - GEE sẽ tự động evaluate khi print
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

// Tạo geometry từ lon, lat
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

// Chuẩn bị features
var features = imageAsset.select(featureNames).clip(studyArea);

// Buffer points để tăng khả năng lấy mẫu
var floodPointsBuffered = floodPoints.map(function(feature) {
  var buffered = feature.buffer(BUFFER_SIZE);
  return buffered.copyProperties(feature, ['flood', 'lat', 'lon']);
});

print('✅ Applied ' + BUFFER_SIZE + 'm buffer to points');

// Sample với buffered points
var trainingData = features.sampleRegions({
  collection: floodPointsBuffered,
  properties: ['flood', 'lat', 'lon'],
  scale: 10,
  tileScale: 1,
  geometries: false
});

print('Total points after sampling:', trainingData.size());

// Kiểm tra null values trong training data
print('\n--- Checking null values in training samples ---');
featureNames.forEach(function(bandName) {
  var countNonNull = trainingData.filter(ee.Filter.notNull([bandName])).size();
  print('  ' + bandName + ':', countNonNull);
});

// Lọc bỏ điểm thiếu dữ liệu
var requiredColumns = ['flood'].concat(featureNames);
trainingData = trainingData.filter(ee.Filter.notNull(requiredColumns));

print('✅ Valid training samples:', trainingData.size());

// Chia train/validation
var withRandom = trainingData.randomColumn('random', 42);
var training = withRandom.filter(ee.Filter.lt('random', TRAIN_SPLIT));
var validation = withRandom.filter(ee.Filter.gte('random', TRAIN_SPLIT));

print('Training samples:', training.size());
print('Validation samples:', validation.size());

//////////////////////////////////////////////////////////////
// BƯỚC 3: TRAIN MODEL
//////////////////////////////////////////////////////////////

print('\n========== STEP 3: TRAINING MODEL ==========');

var rfRegressor = ee.Classifier.smileRandomForest({
  numberOfTrees: 215,
  variablesPerSplit: 6,
  minLeafPopulation: 1,
  bagFraction: 0.5,
  seed: 42
}).setOutputMode('REGRESSION')
  .train({
    features: training,
    classProperty: 'flood',
    inputProperties: featureNames
  });

print('✅ Model trained successfully');

//////////////////////////////////////////////////////////////
// BƯỚC 4: VALIDATION METRICS
//////////////////////////////////////////////////////////////

print('\n========== STEP 4: VALIDATION METRICS ==========');

var validationPredicted = validation.classify(rfRegressor, 'predicted');

// Tính toán errors
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

// Tính RMSE, MAE
var rmse = ee.Number(validationWithErrors.aggregate_mean('squared_error')).sqrt();
var mae = validationWithErrors.aggregate_mean('abs_error');

// Tính R² (coefficient of determination)
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

// Tính Pearson correlation
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

// Feature importance
var importance = rfRegressor.explain();
print('\n--- Feature Importance ---');
print(importance);

//////////////////////////////////////////////////////////////
// BƯỚC 5: PREDICTION
//////////////////////////////////////////////////////////////

print('\n========== STEP 5: MAKING PREDICTIONS ==========');

// Predict flood probability (0-1)
var floodProbability = features.classify(rfRegressor).rename('flood_probability');
floodProbability = floodProbability.clamp(0, 1);

//////////////////////////////////////////////////////////////
// BƯỚC 5.5: KIỂM TRA VÀ NỘI SUY NULL TRONG PREDICTION
//////////////////////////////////////////////////////////////

print('\n========== STEP 5.5: CHECK & INTERPOLATE NULLS IN PREDICTION ==========');

// Thống kê null TRƯỚC nội suy prediction
countNullPixels(floodProbability, ['flood_probability'], studyArea, RESOLUTION * 2,
                'NULL COUNT BEFORE INTERPOLATION (Prediction)');

// Nội suy null values trong prediction
print('\n🔄 Interpolating null values in prediction...');
print('  Radius: 90m (3 pixels) | Iterations: 3');

var floodProbabilityInterpolated = interpolateNulls(
  floodProbability,
  ['flood_probability'],
  90,  // radius 90m
  3    // 3 lần lặp
);

// Clamp lại sau khi nội suy
floodProbabilityInterpolated = floodProbabilityInterpolated.clamp(0, 1);

// Thống kê null SAU nội suy prediction
countNullPixels(floodProbabilityInterpolated, ['flood_probability'], studyArea, RESOLUTION * 2,
                'NULL COUNT AFTER INTERPOLATION (Prediction)');

print('✅ Prediction interpolation completed');

// Classification với threshold
var THRESHOLD = 0.5;
var floodClassification = floodProbabilityInterpolated.gte(THRESHOLD).rename('flood_class');

// Risk levels
var riskLevels = floodProbabilityInterpolated
  .where(floodProbabilityInterpolated.lt(0.25), 1)
  .where(floodProbabilityInterpolated.gte(0.25).and(floodProbabilityInterpolated.lt(0.5)), 2)
  .where(floodProbabilityInterpolated.gte(0.5).and(floodProbabilityInterpolated.lt(0.75)), 3)
  .where(floodProbabilityInterpolated.gte(0.75), 4)
  .rename('risk_level');

// Reproject outputs
floodProbabilityInterpolated = floodProbabilityInterpolated.reproject({
  crs: 'EPSG:4326',
  scale: RESOLUTION
}).clip(studyArea);

floodClassification = floodClassification.reproject({
  crs: 'EPSG:4326',
  scale: RESOLUTION
}).clip(studyArea);

riskLevels = riskLevels.reproject({
  crs: 'EPSG:4326',
  scale: RESOLUTION
}).clip(studyArea);

print('✅ Prediction completed');

// Kiểm tra range
var stats = floodProbabilityInterpolated.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: studyArea,
  scale: RESOLUTION * 10,
  maxPixels: 1e9
});
print('Probability range - Min:', stats.get('flood_probability_min'), '| Max:', stats.get('flood_probability_max'));

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

// Flood probability (continuous) - INTERPOLATED VERSION
var probPalette = {
  min: 0,
  max: 1,
  palette: ['#00FF00', '#FFFF00', '#FF9900', '#FF0000']
};
Map.addLayer(floodProbabilityInterpolated, probPalette, '🎯 Flood Probability (Interpolated)', true);

// Classification (binary)
var classPalette = {
  min: 0,
  max: 1,
  palette: ['green', 'red']
};
Map.addLayer(floodClassification, classPalette, 'Classification (Threshold=0.5)', false);

// Risk levels
var riskPalette = {
  min: 1,
  max: 4,
  palette: ['#00FF00', '#FFFF00', '#FF9900', '#FF0000']
};
Map.addLayer(riskLevels, riskPalette, 'Risk Levels', false);

// Legend
var legend = ui.Panel({
  style: {position: 'bottom-left', padding: '8px 15px'}
});

var legendTitle = ui.Label({
  value: 'Flood Probability',
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

legend.add(makeRow('#00FF00', '0.00 - 0.25: Low Risk'));
legend.add(makeRow('#FFFF00', '0.25 - 0.50: Moderate Risk'));
legend.add(makeRow('#FF9900', '0.50 - 0.75: High Risk'));
legend.add(makeRow('#FF0000', '0.75 - 1.00: Very High Risk'));

Map.add(legend);

print('✅ Map layers added');

//////////////////////////////////////////////////////////////
// BƯỚC 7: EXPORT
//////////////////////////////////////////////////////////////

print('\n========== STEP 7: EXPORTING RESULTS ==========');

// Export probability map (INTERPOLATED)
Export.image.toDrive({
  image: floodProbabilityInterpolated,
  description: 'flood_probability_RF_interpolated',
  scale: RESOLUTION,
  region: studyArea,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// Export classification map
Export.image.toDrive({
  image: floodClassification,
  description: 'flood_classification_RF_interpolated',
  scale: RESOLUTION,
  region: studyArea,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// Export risk levels
Export.image.toDrive({
  image: riskLevels,
  description: 'flood_risk_levels_RF_interpolated',
  scale: RESOLUTION,
  region: studyArea,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// Export validation results
Export.table.toDrive({
  collection: validationWithErrors.select(['lat', 'lon', 'flood', 'predicted', 'error', 'squared_error', 'abs_error']),
  description: 'validation_results_RF_interpolated',
  fileFormat: 'CSV'
});

print('\n========== ✅ PROCESS COMPLETED ==========');
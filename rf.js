// ========================================================================
// FLOOD RISK PREDICTION MODEL USING RANDOM FOREST REGRESSION
// Bài toán: HỒI QUY - Dự đoán xác suất liên tục từ 0-1
// Luồng: Load Assets → Extract Features → Train Model → Predict
// ========================================================================

// CẤU HÌNH MÔ HÌNH
var RESOLUTION = 30;  // 30m resolution
var NUM_TREES = 200;   // Số cây trong Random Forest
var TILE_SCALE = 16;  // Tăng tốc xử lý
var TRAIN_SPLIT = 0.7; // 70% train, 30% validation

// Tên 13 features từ imageAsset
var featureNames = [
  'lulc', 'Density_River', 'Density_Road', 'Distan2river', 'Distan2road_met',
  'aspect', 'curvature', 'dem', 'flowDir', 'slope', 'twi', 'NDVI', 'rainfall'
];

print('========== FLOOD RISK PREDICTION - RANDOM FOREST REGRESSION ==========');
print('Problem Type: REGRESSION (continuous probability 0-1)');
print('Resolution: ' + RESOLUTION + 'm');
print('Number of trees: ' + NUM_TREES);
print('Features: 13 bands from imageAsset');

//////////////////////////////////////////////////////////////
// BƯỚC 1: LOAD 3 FILE ASSETS ĐẦU VÀO
//////////////////////////////////////////////////////////////

print('========== STEP 1: LOADING INPUT ASSETS ==========');

// 1.1 Load Study Area (SHP file)
if (typeof studyArea === 'undefined') {
  print('ERROR: studyArea chưa được import!');
  print('Hướng dẫn: Import SHP file từ Assets tab và đặt tên là "studyArea"');
}
print('✅ Study area loaded');

// 1.2 Load Image Features (13 bands)
if (typeof imageAsset === 'undefined') {
  print('ERROR: imageAsset chưa được import!');
  print('Hướng dẫn: Import Image asset (13 bands) và đặt tên là "imageAsset"');
}
print('✅ Image asset loaded');

// 1.3 Load Flood Points (CSV với 3 cột: lat, lon, flood)
if (typeof floodPoints === 'undefined') {
  print('ERROR: floodPoints chưa được import!');
  print('Hướng dẫn: Import CSV file (lat, lon, flood) và đặt tên là "floodPoints"');
}

// Đảm bảo geometry được tạo đúng từ lon, lat
var floodPoints = floodPoints.map(function(feature) {
  var lon = ee.Number(feature.get('lon'));
  var lat = ee.Number(feature.get('lat'));
  // Đảm bảo flood value là số (0 hoặc 1)
  var floodValue = ee.Number(feature.get('flood'));
  return feature.set('flood', floodValue)
                .setGeometry(ee.Geometry.Point([lon, lat], 'EPSG:4326'));
});

print('✅ Flood points loaded and geometries created');

// Kiểm tra cấu trúc flood points
var samplePoint = ee.Feature(floodPoints.first());
print('Sample flood point properties:', samplePoint.propertyNames());
print('Sample flood value:', samplePoint.get('flood'));
print('Total flood points:', floodPoints.size());

// Đếm số điểm flood và non-flood
var floodCount = floodPoints.filter(ee.Filter.eq('flood', 1)).size();
var nonFloodCount = floodPoints.filter(ee.Filter.eq('flood', 0)).size();
print('Flood points (flood=1):', floodCount);
print('Non-flood points (flood=0):', nonFloodCount);

//////////////////////////////////////////////////////////////
// BƯỚC 2: TRÍCH XUẤT 13 FEATURES TẠI CÁC ĐIỂM NHÃN
//////////////////////////////////////////////////////////////

print('========== STEP 2: EXTRACTING FEATURES AT LABELED POINTS ==========');

// 2.0 DEBUG: Kiểm tra band names
print('Expected feature names:', featureNames);
print('Actual band names in imageAsset:', imageAsset.bandNames());

// 2.1 Chuẩn bị features từ imageAsset
var features = imageAsset.select(featureNames).clip(studyArea);

// Reproject với scale mong muốn
features = features.reproject({
  crs: 'EPSG:4326',
  scale: RESOLUTION
});

print('✅ Features prepared with', featureNames.length, 'bands');

// 2.1.5 DEBUG: Test trên 1 điểm cụ thể
print('========== DEBUG: TESTING FIRST POINT ==========');
var firstPoint = ee.Feature(floodPoints.first());
print('First point properties:', firstPoint.toDictionary());

var testGeom = firstPoint.geometry();
print('Test geometry coordinates:', testGeom.coordinates());

// Lấy giá trị tại điểm này
var pixelValues = features.reduceRegion({
  reducer: ee.Reducer.first(),
  geometry: testGeom,
  scale: RESOLUTION,
  maxPixels: 1e9
});
print('🎯 Pixel values at first point:', pixelValues);
print('🎯 Number of non-null values:', pixelValues.keys().length());

// 2.2 Sample features tại các điểm flood
print('========== SAMPLING ALL POINTS ==========');

var trainingData = features.sampleRegions({
  collection: floodPoints,
  properties: ['flood', 'lat', 'lon'],
  scale: RESOLUTION,
  tileScale: TILE_SCALE,
  geometries: true  // Giữ geometry để debug
});

print('Total points after sampling:', trainingData.size());

// DEBUG: Xem 3 điểm đầu tiên sau khi sample
var firstThree = trainingData.limit(3);
print('First 3 sampled points (check for null values):', firstThree);

// Kiểm tra từng feature xem feature nào bị null
print('========== CHECKING NULL VALUES PER FEATURE ==========');
featureNames.forEach(function(bandName) {
  var countNonNull = trainingData.filter(ee.Filter.notNull([bandName])).size();
  print('  Feature "' + bandName + '" - Non-null points:', countNonNull);
});

// 2.3 Lọc bỏ các điểm thiếu dữ liệu
var requiredColumns = ['flood'].concat(featureNames);
trainingData = trainingData.filter(ee.Filter.notNull(requiredColumns));

print('✅ Valid training samples after filtering:', trainingData.size());

// Nếu không có training data, dừng lại và hiển thị visualization
var validSampleCount = trainingData.size();
validSampleCount.evaluate(function(count) {
  if (count === 0) {
    print('❌ ERROR: NO VALID TRAINING DATA!');
    print('Possible reasons:');
    print('  1. Band names mismatch - Check "Expected" vs "Actual" band names above');
    print('  2. Points outside imageAsset coverage - Check map visualization');
    print('  3. NoData values in imageAsset at point locations');
    print('');
    print('ACTION REQUIRED:');
    print('  → Check the map to see if points overlap with image data');
    print('  → Verify band names match exactly (case-sensitive)');
    print('  → Check if imageAsset has valid data in study area');
  }
});

// 2.4 Chia dữ liệu train/validation
var withRandom = trainingData.randomColumn('random', 42);
var training = withRandom.filter(ee.Filter.lt('random', TRAIN_SPLIT));
var validation = withRandom.filter(ee.Filter.gte('random', TRAIN_SPLIT));

print('Training samples:', training.size());
print('Validation samples:', validation.size());

//////////////////////////////////////////////////////////////
// BƯỚC 3: HUẤN LUYỆN MÔ HÌNH RANDOM FOREST REGRESSION
//////////////////////////////////////////////////////////////

print('========== STEP 3: TRAINING RANDOM FOREST REGRESSOR ==========');

// SỬ DỤNG REGRESSOR THAY VÌ CLASSIFIER
var rfRegressor = ee.Classifier.smileRandomForest({
  numberOfTrees: NUM_TREES,
  variablesPerSplit: null,      // sqrt(numFeatures) - tự động
  minLeafPopulation: 1,
  bagFraction: 0.5,
  seed: 42
}).setOutputMode('REGRESSION')  // QUAN TRỌNG: Chế độ REGRESSION
  .train({
    features: training,
    classProperty: 'flood',  // Dù tên là classProperty nhưng đây là giá trị liên tục
    inputProperties: featureNames
  });

print('✅ Random Forest Regressor trained successfully');
print('Model configuration:');
print('  - Mode: REGRESSION (continuous output)');
print('  - Trees:', NUM_TREES);
print('  - Input features:', featureNames.length);

// 3.1 Đánh giá trên tập validation
print('========== VALIDATION METRICS (REGRESSION) ==========');

var validationPredicted = validation.classify(rfRegressor, 'predicted');

// Tính các metrics cho regression
var validationMetrics = validationPredicted.select(['flood', 'predicted'])
  .reduceColumns({
    reducer: ee.Reducer.spearmansCorrelation().combine({
      reducer2: ee.Reducer.pearsonsCorrelation(),
      sharedInputs: true
    }),
    selectors: ['flood', 'predicted']
  });

print('Spearman Correlation:', validationMetrics.get('correlation'));
print('Pearson Correlation:', validationMetrics.get('correlation1'));

// Tính RMSE và MAE
var errors = validationPredicted.map(function(f) {
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

var rmse = errors.aggregate_mean('squared_error');
var mae = errors.aggregate_mean('abs_error');

print('RMSE (Root Mean Squared Error):', rmse);
print('MAE (Mean Absolute Error):', mae);

// Feature importance
var importance = rfRegressor.explain();
print('Feature Importance:', importance);

//////////////////////////////////////////////////////////////
// BƯỚC 4: THỰC HIỆN PREDICT TRÊN TOÀN BỘ KHU VỰC
//////////////////////////////////////////////////////////////

print('========== STEP 4: MAKING PREDICTIONS (REGRESSION) ==========');

// 4.1 Predict flood probability (liên tục 0-1) cho toàn bộ khu vực
var floodProbability = features.classify(rfRegressor).rename('flood_probability');

// Đảm bảo giá trị trong khoảng [0, 1]
floodProbability = floodProbability.clamp(0, 1);

// 4.2 Tạo bản đồ phân loại dựa trên ngưỡng (optional)
var THRESHOLD = 0.5;  // Ngưỡng phân loại: >= 0.5 là flood
var floodClassification = floodProbability.gte(THRESHOLD).rename('flood_class');

// 4.3 Tạo bản đồ risk levels
var riskLevels = floodProbability
  .where(floodProbability.lt(0.25), 1)  // Low risk
  .where(floodProbability.gte(0.25).and(floodProbability.lt(0.5)), 2)  // Moderate
  .where(floodProbability.gte(0.5).and(floodProbability.lt(0.75)), 3)  // High
  .where(floodProbability.gte(0.75), 4)  // Very High
  .rename('risk_level');

// 4.4 Reproject kết quả
floodProbability = floodProbability.reproject({
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
print('Output range check - Min/Max probability:');
var stats = floodProbability.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: studyArea,
  scale: RESOLUTION * 10,  // Coarser scale for speed
  maxPixels: 1e9
});
print('  Min probability:', stats.get('flood_probability_min'));
print('  Max probability:', stats.get('flood_probability_max'));

//////////////////////////////////////////////////////////////
// BƯỚC 5: HIỂN THỊ KẾT QUẢ
//////////////////////////////////////////////////////////////

print('========== STEP 5: DISPLAYING RESULTS ==========');

// Hiển thị bản đồ
Map.centerObject(studyArea, 10);

// Hiển thị khu vực nghiên cứu
Map.addLayer(studyArea, {color: '0000FF'}, 'Study Area Boundary', true, 0.3);

// Hiển thị imageAsset để kiểm tra coverage
Map.addLayer(imageAsset.select(0), {min: 0, max: 1000}, 'Image Asset (First Band)', false);

// Hiển thị các điểm training
var floodPts = floodPoints.filter(ee.Filter.eq('flood', 1));
var nonFloodPts = floodPoints.filter(ee.Filter.eq('flood', 0));
Map.addLayer(floodPts, {color: 'FF0000'}, 'Flood Points (1)', true);
Map.addLayer(nonFloodPts, {color: '00FF00'}, 'Non-Flood Points (0)', true);

// Hiển thị kết quả REGRESSION - Continuous Probability (0-1)
var probPalette = {
  min: 0,
  max: 1,
  palette: ['#00FF00', '#FFFF00', '#FF9900', '#FF0000']  // Green → Yellow → Orange → Red
};
Map.addLayer(floodProbability, probPalette, '🎯 Flood Probability (Continuous 0-1)', true);

// Hiển thị classification với ngưỡng (optional)
var classPalette = {
  min: 0,
  max: 1,
  palette: ['green', 'red']
};
Map.addLayer(floodClassification, classPalette, 'Flood Classification (Threshold=0.5)', false);

// Hiển thị risk levels
var riskPalette = {
  min: 1,
  max: 4,
  palette: ['#00FF00', '#FFFF00', '#FF9900', '#FF0000']
};
Map.addLayer(riskLevels, riskPalette, 'Risk Levels (Low/Mod/High/VeryHigh)', false);

// Thêm legend
var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px'
  }
});

var legendTitle = ui.Label({
  value: 'Flood Probability',
  style: {fontWeight: 'bold', fontSize: '16px', margin: '0 0 4px 0'}
});
legend.add(legendTitle);

var makeRow = function(color, name) {
  var colorBox = ui.Label({
    style: {
      backgroundColor: color,
      padding: '8px',
      margin: '0 0 4px 0'
    }
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

print('✅ Map layers added - Check visualization');

//////////////////////////////////////////////////////////////
// BƯỚC 6: EXPORT KẾT QUẢ
//////////////////////////////////////////////////////////////

print('========== STEP 6: EXPORTING RESULTS ==========');

// Export flood probability map (CONTINUOUS 0-1)
Export.image.toDrive({
  image: floodProbability,
  description: 'flood_probability_RF_regression',
  scale: RESOLUTION,
  region: studyArea,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// Export flood classification map (binary with threshold)
Export.image.toDrive({
  image: floodClassification,
  description: 'flood_classification_RF_threshold',
  scale: RESOLUTION,
  region: studyArea,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// Export risk levels map
Export.image.toDrive({
  image: riskLevels,
  description: 'flood_risk_levels_RF',
  scale: RESOLUTION,
  region: studyArea,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// Export validation results với predicted values
Export.table.toDrive({
  collection: validationPredicted.select(['lat', 'lon', 'flood', 'predicted', 'error', 'squared_error', 'abs_error']),
  description: 'validation_results_RF_regression',
  fileFormat: 'CSV'
});

print('========== PROCESS COMPLETED ==========');
print('✅ Check Tasks tab to run exports');
print('');
print('📊 REGRESSION OUTPUTS:');
print('  1. Flood Probability Map (continuous 0-1)');
print('  2. Flood Classification Map (binary with threshold=0.5)');
print('  3. Risk Levels Map (4 categories)');
print('  4. Validation Results CSV (with predicted values)');
print('');
print('⚠️ If you see "No valid training data" error:');
print('  1. Check console logs above for band name mismatches');
print('  2. Verify points overlap with image data on map');
print('  3. Ensure imageAsset has valid data (not all NoData)');
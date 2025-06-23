package es.albertosoto.spark_dtree.controller;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Collections;

/**
 * Controller for Decision Tree operations
 */
@RestController
@RequestMapping("/api/dtree")
public class DecisionTreeController {
    private SparkSession spark;
    private PipelineModel model;

    public DecisionTreeController() {
        // Initialize Spark session
        this.spark = SparkSession.builder()
                .appName("SparkDecisionTree")
                .master("local[*]")
                .getOrCreate();
    }

    /**
     * Get information about the Decision Tree service
     * @return Service information
     */
    @GetMapping("/info")
    public ResponseEntity<Map<String, Object>> getInfo() {
        Map<String, Object> info = new HashMap<>();
        info.put("service", "Spark Decision Tree API");
        info.put("version", "1.0.0");
        info.put("description", "API for working with Apache Spark Decision Trees");
        info.put("status", "Ready");

        return ResponseEntity.ok(info);
    }

    /**
     * Train a decision tree model using the Iris dataset
     * @return Training results
     */
    @PostMapping("/train")
    public ResponseEntity<Map<String, Object>> trainModel() {
        Map<String, Object> result = new HashMap<>();

        try {
            // Load the Iris dataset (assuming it's in CSV format)
            Dataset<Row> data = spark.read()
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .csv("src/main/resources/data/iris.csv");

            // Split the data into training and test sets
            Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3}, 1234L);
            Dataset<Row> trainingData = splits[0];
            Dataset<Row> testData = splits[1];

            // Define feature columns
            String[] featureColumns = {"sepal_length", "sepal_width", "petal_length", "petal_width"};

            // Create feature vector
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(featureColumns)
                    .setOutputCol("features");

            // Index the label column - don't call fit() here
            StringIndexer labelIndexer = new StringIndexer()
                    .setInputCol("species")
                    .setOutputCol("indexedLabel")
                    .setHandleInvalid("skip");

            // Create and configure the decision tree model
            DecisionTreeClassifier dt = new DecisionTreeClassifier()
                    .setLabelCol("indexedLabel")
                    .setFeaturesCol("features")
                    .setMaxDepth(5)
                    .setImpurity("gini");

            // Create the pipeline
            Pipeline pipeline = new Pipeline()
                    .setStages(new PipelineStage[]{labelIndexer, assembler, dt});

            // Train the model
            this.model = pipeline.fit(trainingData);

            // Make predictions on test data
            Dataset<Row> predictions = model.transform(testData);

            // Evaluate the model
            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                    .setLabelCol("indexedLabel")
                    .setPredictionCol("prediction")
                    .setMetricName("accuracy");
            double accuracy = evaluator.evaluate(predictions);

            // Return results
            result.put("status", "success");
            result.put("message", "Decision tree model trained successfully");
            result.put("accuracy", accuracy);
            result.put("numTrainingSamples", trainingData.count());
            result.put("numTestSamples", testData.count());

            return ResponseEntity.ok(result);
        } catch (Exception e) {
            result.put("status", "error");
            result.put("message", "Error training model: " + e.getMessage());
            return ResponseEntity.status(500).body(result);
        }
    }

    /**
     * Make a prediction using the trained model
     * @param features Map containing feature values
     * @return Prediction result
     */
    @PostMapping("/predict")
    public ResponseEntity<Map<String, Object>> predict(@RequestBody Map<String, Double> features) {
        Map<String, Object> result = new HashMap<>();

        if (model == null) {
            result.put("status", "error");
            result.put("message", "Model not trained yet. Please train the model first.");
            return ResponseEntity.status(400).body(result);
        }

        try {
            // Extract feature values
            Double sepalLength = features.get("sepal_length");
            Double sepalWidth = features.get("sepal_width");
            Double petalLength = features.get("petal_length");
            Double petalWidth = features.get("petal_width");

            // Create a Row object with the feature values
            Row row = RowFactory.create(sepalLength, sepalWidth, petalLength, petalWidth);

            // Create a list of rows
            List<Row> rows = Collections.singletonList(row);

            // Create schema for the input data
            StructType schema = DataTypes.createStructType(new StructField[] {
                DataTypes.createStructField("sepal_length", DataTypes.DoubleType, true),
                DataTypes.createStructField("sepal_width", DataTypes.DoubleType, true),
                DataTypes.createStructField("petal_length", DataTypes.DoubleType, true),
                DataTypes.createStructField("petal_width", DataTypes.DoubleType, true)
            });

            // Create DataFrame from input data
            Dataset<Row> inputData = spark.createDataFrame(rows, schema);

            // Make prediction
            Dataset<Row> prediction = model.transform(inputData);

            // Extract prediction result
            double predictedClass = prediction.select("prediction").first().getDouble(0);

            // Return result
            result.put("status", "success");
            result.put("prediction", predictedClass);
            result.put("features", features);

            return ResponseEntity.ok(result);
        } catch (Exception e) {
            result.put("status", "error");
            result.put("message", "Error making prediction: " + e.getMessage());
            return ResponseEntity.status(500).body(result);
        }
    }

    /**
     * Get model information
     * @return Model information
     */
    @GetMapping("/model")
    public ResponseEntity<Map<String, Object>> getModelInfo() {
        Map<String, Object> info = new HashMap<>();

        if (model == null) {
            info.put("status", "not_trained");
            info.put("message", "Model not trained yet");
        } else {
            info.put("status", "trained");
            info.put("message", "Model is trained and ready for predictions");

            try {
                // Extract decision tree model from pipeline model
                // The DecisionTreeClassificationModel is at index 2 in the pipeline stages
                DecisionTreeClassificationModel dtModel = (DecisionTreeClassificationModel) model.stages()[2];

                info.put("modelType", "DecisionTreeClassifier");
                info.put("maxDepth", dtModel.depth());
                info.put("numNodes", dtModel.numNodes());
                info.put("featureImportances", dtModel.featureImportances().toString());
            } catch (Exception e) {
                info.put("modelDetails", "Could not extract model details: " + e.getMessage());
            }
        }

        return ResponseEntity.ok(info);
    }
}

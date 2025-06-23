package es.albertosoto.spark_dtree.service;

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
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Alberto Soto. 23/6/25
 */
@Service
public class SparkService {

    private final SparkSession sparkSession;
    private PipelineModel model;

    @Autowired
    public SparkService(SparkSession sparkSession) {
        this.sparkSession = sparkSession;
    }

    /**
     * Train a decision tree model using the Iris dataset
     * @return Training results
     */
    public Map<String, Object> trainModel() {
        Map<String, Object> result = new HashMap<>();

        try {
            // Load the Iris dataset
            Dataset<Row> data = sparkSession.read()
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .csv(getClass().getClassLoader().getResource("data/iris.csv").getPath());

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

            // Index the label column
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

            return result;
        } catch (Exception e) {
            result.put("status", "error");
            result.put("message", "Error training model: " + e.getMessage());
            return result;
        }
    }

    /**
     * Make a prediction using the trained model
     * @param features Map containing feature values
     * @return Prediction result
     */
    public Map<String, Object> predict(Map<String, Double> features) {
        Map<String, Object> result = new HashMap<>();

        if (model == null) {
            result.put("status", "error");
            result.put("message", "Model not trained yet. Please train the model first.");
            return result;
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
            Dataset<Row> inputData = sparkSession.createDataFrame(rows, schema);

            // Make prediction
            Dataset<Row> prediction = model.transform(inputData);

            // Extract prediction result
            double predictedClass = prediction.select("prediction").first().getDouble(0);

            // Return result
            result.put("status", "success");
            result.put("prediction", predictedClass);
            result.put("features", features);

            return result;
        } catch (Exception e) {
            result.put("status", "error");
            result.put("message", "Error making prediction: " + e.getMessage());
            return result;
        }
    }

    /**
     * Get model information
     * @return Model information
     */
    public Map<String, Object> getModelInfo() {
        Map<String, Object> info = new HashMap<>();

        if (model == null) {
            info.put("status", "not_trained");
            info.put("message", "Model not trained yet");
        } else {
            info.put("status", "trained");
            info.put("message", "Model is trained and ready for predictions");

            try {
                // Extract decision tree model from pipeline model
                DecisionTreeClassificationModel dtModel = (DecisionTreeClassificationModel) model.stages()[2];

                info.put("modelType", "DecisionTreeClassifier");
                info.put("maxDepth", dtModel.depth());
                info.put("numNodes", dtModel.numNodes());
                info.put("featureImportances", dtModel.featureImportances().toString());
            } catch (Exception e) {
                info.put("modelDetails", "Could not extract model details: " + e.getMessage());
            }
        }

        return info;
    }
}

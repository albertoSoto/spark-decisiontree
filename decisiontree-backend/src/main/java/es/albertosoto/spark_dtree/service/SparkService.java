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

import java.util.ArrayList;
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

    /**
     * Extract the decision tree structure for frontend rendering
     * @return Tree structure in a format suitable for frontend visualization
     */
    public Map<String, Object> getTreeStructure() {
        Map<String, Object> result = new HashMap<>();

        if (model == null) {
            result.put("status", "not_trained");
            result.put("message", "Model not trained yet");
            return result;
        }

        try {
            // Extract decision tree model from pipeline model
            DecisionTreeClassificationModel dtModel = (DecisionTreeClassificationModel) model.stages()[2];
            
            // Get the feature names
            String[] featureNames = {"sepal_length", "sepal_width", "petal_length", "petal_width"};
            
            // Get the tree as a string representation
            String treeString = dtModel.toDebugString();
            
            // Parse the tree string into a structured format
            Map<String, Object> treeStructure = parseTreeString(treeString, featureNames);
            
            result.put("status", "success");
            result.put("treeStructure", treeStructure);
            result.put("rawTree", treeString);
            
            // Add basic model information
            result.put("numNodes", dtModel.numNodes());
            result.put("depth", dtModel.depth());
            result.put("featureImportances", dtModel.featureImportances().toString());
            
            return result;
        } catch (Exception e) {
            result.put("status", "error");
            result.put("message", "Error extracting tree structure: " + e.getMessage());
            result.put("stackTrace", e.getStackTrace());
            return result;
        }
    }
    
    /**
     * Parse the decision tree string representation into a structured format
     * @param treeString The string representation of the tree
     * @param featureNames Array of feature names
     * @return Hierarchical map representing the tree structure
     */
    private Map<String, Object> parseTreeString(String treeString, String[] featureNames) {
        Map<String, Object> root = new HashMap<>();
        
        // Split the tree string into lines
        String[] lines = treeString.split("\n");
        
        // Extract the root node information from the first line
        // Example: "DecisionTreeClassificationModel (uid=dtc_abc123) of depth 5 with 31 nodes"
        String firstLine = lines[0];
        root.put("modelType", firstLine.split(" ")[0]);
        
        // Extract depth from the first line
        int depth = 0;
        for (String part : firstLine.split(" ")) {
            if (part.equals("depth")) {
                String depthStr = firstLine.split("depth ")[1].split(" ")[0];
                depth = Integer.parseInt(depthStr);
                break;
            }
        }
        root.put("depth", depth);
        
        // Extract number of nodes from the first line
        int numNodes = 0;
        for (String part : firstLine.split(" ")) {
            if (part.equals("with")) {
                String nodesStr = firstLine.split("with ")[1].split(" ")[0];
                numNodes = Integer.parseInt(nodesStr);
                break;
            }
        }
        root.put("numNodes", numNodes);
        
        // Parse the tree structure from the remaining lines
        Map<String, Object> treeNodes = new HashMap<>();
        List<Map<String, Object>> nodesList = new ArrayList<>();
        
        // Skip the first line as it contains the model summary
        for (int i = 1; i < lines.length; i++) {
            String line = lines[i];
            
            // Skip empty lines
            if (line.trim().isEmpty()) {
                continue;
            }
            
            Map<String, Object> node = new HashMap<>();
            
            // Calculate the depth of this node based on indentation
            int nodeDepth = 0;
            int j = 0;
            while (j < line.length() && line.charAt(j) == ' ') {
                nodeDepth++;
                j++;
            }
            nodeDepth = nodeDepth / 2; // Assuming 2 spaces per level
            
            node.put("depth", nodeDepth);
            
            // Clean up the line by removing leading spaces
            line = line.trim();
            
            // Check if this is a split node or a leaf node
            if (line.startsWith("If (")) {
                // This is a split node
                node.put("type", "split");
                
                // Extract feature index and threshold
                // Example: "If (feature 2 <= 2.45)"
                String featureStr = line.substring(line.indexOf("feature ") + 8, line.indexOf(" <= "));
                String thresholdStr = line.substring(line.indexOf(" <= ") + 4, line.indexOf(")"));
                
                int featureIndex = Integer.parseInt(featureStr);
                double threshold = Double.parseDouble(thresholdStr);
                
                node.put("feature", featureIndex);
                node.put("threshold", threshold);
                
                // Add feature name if available
                if (featureNames != null && featureIndex < featureNames.length) {
                    node.put("featureName", featureNames[featureIndex]);
                    node.put("splitCondition", featureNames[featureIndex] + " <= " + threshold);
                } else {
                    node.put("featureName", "feature_" + featureIndex);
                    node.put("splitCondition", "feature_" + featureIndex + " <= " + threshold);
                }
                
            } else if (line.startsWith("Predict:")) {
                // This is a leaf node
                node.put("type", "leaf");
                
                // Extract prediction value
                // Example: "Predict: 0.0"
                String predictionStr = line.substring(line.indexOf("Predict: ") + 9).trim();
                double prediction = Double.parseDouble(predictionStr);
                
                node.put("prediction", prediction);
                node.put("predictionClass", String.valueOf((int)prediction));
            } else if (line.startsWith("Else")) {
                // This is an "Else" condition, we'll skip it as it's handled implicitly
                continue;
            }
            
            // Add node ID based on position in the list
            node.put("nodeId", nodesList.size());
            
            // Add the node to our list
            nodesList.add(node);
        }
        
        // Build the tree structure by connecting parent-child relationships
        buildTreeRelationships(nodesList);
        
        // The first node in the list is the root node of the tree
        if (!nodesList.isEmpty()) {
            root.put("rootNode", nodesList.get(0));
        }
        
        return root;
    }
    
    /**
     * Build parent-child relationships between nodes based on their depth
     * @param nodesList List of nodes in pre-order traversal
     */
    private void buildTreeRelationships(List<Map<String, Object>> nodesList) {
        if (nodesList.isEmpty()) {
            return;
        }
        
        // Stack to keep track of parent nodes at each depth
        Map<Integer, Map<String, Object>> parentStack = new HashMap<>();
        
        // Process each node
        for (Map<String, Object> node : nodesList) {
            int depth = (int) node.get("depth");
            
            // If this is not the root node, connect it to its parent
            if (depth > 0) {
                Map<String, Object> parent = parentStack.get(depth - 1);
                
                if (parent != null) {
                    // Check if this is the first or second child of the parent
                    if (!parent.containsKey("leftChild")) {
                        parent.put("leftChild", node);
                    } else if (!parent.containsKey("rightChild")) {
                        parent.put("rightChild", node);
                    }
                }
            }
            
            // If this is a split node, add it to the parent stack
            if ("split".equals(node.get("type"))) {
                parentStack.put(depth, node);
            }
        }
    }
}

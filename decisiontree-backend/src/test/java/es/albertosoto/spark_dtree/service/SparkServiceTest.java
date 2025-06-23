package es.albertosoto.spark_dtree.service;

import es.albertosoto.spark_dtree.util.SparkTestUtils;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for SparkService using a real SparkSession in local mode
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class SparkServiceTest {

    private SparkSession sparkSession;
    private SparkService sparkService;
    private Dataset<Row> testData;

    @BeforeAll
    public void setup() throws IOException {
        // Create a test SparkSession
        sparkSession = SparkTestUtils.createTestSparkSession();
        
        // Create the SparkService with the test SparkSession
        sparkService = new SparkService(sparkSession);
        
        // Create sample test data
        testData = SparkTestUtils.createSampleIrisDataset(sparkSession);
        
        // Register the test data as a temp view so we can use it in tests
        testData.createOrReplaceTempView("iris_test_data");
    }
    
    @AfterAll
    public void tearDown() throws IOException {
        // Close the SparkSession
        if (sparkSession != null) {
            sparkSession.close();
        }
    }
    
    @Test
    public void testTrainModel_ModelNotNull() {
        // Create a test directory with iris.csv
        try {
            Path tempDir = Files.createTempDirectory("spark-test");
            Path dataDir = Paths.get(tempDir.toString(), "src", "main", "resources", "data");
            Files.createDirectories(dataDir);
            
            // Create a simple iris.csv file
            Path csvPath = Paths.get(dataDir.toString(), "iris.csv");
            String csvContent = "sepal_length,sepal_width,petal_length,petal_width,species\n" +
                "5.1,3.5,1.4,0.2,setosa\n" +
                "7.0,3.2,4.7,1.4,versicolor\n" +
                "6.3,3.3,6.0,2.5,virginica\n";
            Files.write(csvPath, csvContent.getBytes(StandardCharsets.UTF_8));
            
            // Set system property to point to the temp directory for file loading
            String originalDir = System.getProperty("user.dir");
            System.setProperty("user.dir", tempDir.toString());
            
            try {
                // Train the model
                Map<String, Object> result = sparkService.trainModel();
                
                // Verify
                assertNotNull(result);
                assertEquals("success", result.get("status"));
                assertTrue(result.containsKey("accuracy"));
                assertTrue(result.containsKey("numTrainingSamples"));
                assertTrue(result.containsKey("numTestSamples"));
            } finally {
                // Restore original directory
                System.setProperty("user.dir", originalDir);
                
                // Clean up temp directory
                deleteDirectory(tempDir.toFile());
            }
        } catch (IOException e) {
            fail("Failed to create test data: " + e.getMessage());
        }
    }
    
    @Test
    public void testPredict_ModelNotTrained() {
        // Create a new SparkService to ensure model is null
        SparkService untrained = new SparkService(sparkSession);
        
        Map<String, Double> features = new HashMap<>();
        features.put("sepal_length", 5.1);
        features.put("sepal_width", 3.5);
        features.put("petal_length", 1.4);
        features.put("petal_width", 0.2);
        
        Map<String, Object> result = untrained.predict(features);
        
        assertNotNull(result);
        assertEquals("error", result.get("status"));
        assertEquals("Model not trained yet. Please train the model first.", result.get("message"));
    }
    
    @Test
    public void testGetModelInfo_ModelNotTrained() {
        // Create a new SparkService to ensure model is null
        SparkService untrained = new SparkService(sparkSession);
        
        Map<String, Object> result = untrained.getModelInfo();
        
        assertNotNull(result);
        assertEquals("not_trained", result.get("status"));
        assertEquals("Model not trained yet", result.get("message"));
    }
    
    @Test
    public void testEndToEndWorkflow() {
        // Create a test directory with iris.csv
        try {
            Path tempDir = Files.createTempDirectory("spark-test");
            Path dataDir = Paths.get(tempDir.toString(), "src", "main", "resources", "data");
            Files.createDirectories(dataDir);
            
            // Create a simple iris.csv file
            Path csvPath = Paths.get(dataDir.toString(), "iris.csv");
            String csvContent = "sepal_length,sepal_width,petal_length,petal_width,species\n" +
                "5.1,3.5,1.4,0.2,setosa\n" +
                "4.9,3.0,1.4,0.2,setosa\n" +
                "4.7,3.2,1.3,0.2,setosa\n" +
                "7.0,3.2,4.7,1.4,versicolor\n" +
                "6.4,3.2,4.5,1.5,versicolor\n" +
                "6.9,3.1,4.9,1.5,versicolor\n" +
                "6.3,3.3,6.0,2.5,virginica\n" +
                "5.8,2.7,5.1,1.9,virginica\n" +
                "7.1,3.0,5.9,2.1,virginica\n";
            Files.write(csvPath, csvContent.getBytes(StandardCharsets.UTF_8));
            
            // Set system property to point to the temp directory for file loading
            String originalDir = System.getProperty("user.dir");
            System.setProperty("user.dir", tempDir.toString());
            
            try {
                // Create a new service instance for this test
                SparkService service = new SparkService(sparkSession);
                
                // Train the model
                Map<String, Object> trainResult = service.trainModel();
                assertNotNull(trainResult);
                assertEquals("success", trainResult.get("status"));
                
                // Get model info
                Map<String, Object> modelInfo = service.getModelInfo();
                assertNotNull(modelInfo);
                assertEquals("trained", modelInfo.get("status"));
                
                // Make a prediction
                Map<String, Double> features = new HashMap<>();
                features.put("sepal_length", 5.1);
                features.put("sepal_width", 3.5);
                features.put("petal_length", 1.4);
                features.put("petal_width", 0.2);
                
                Map<String, Object> prediction = service.predict(features);
                assertNotNull(prediction);
                assertEquals("success", prediction.get("status"));
                assertNotNull(prediction.get("prediction"));
            } finally {
                // Restore original directory
                System.setProperty("user.dir", originalDir);
                
                // Clean up temp directory
                deleteDirectory(tempDir.toFile());
            }
        } catch (IOException e) {
            fail("Failed to create test data: " + e.getMessage());
        }
    }
    
    // Helper method to recursively delete a directory
    private void deleteDirectory(File directory) {
        if (directory.exists()) {
            File[] files = directory.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isDirectory()) {
                        deleteDirectory(file);
                    } else {
                        file.delete();
                    }
                }
            }
            directory.delete();
        }
    }
}

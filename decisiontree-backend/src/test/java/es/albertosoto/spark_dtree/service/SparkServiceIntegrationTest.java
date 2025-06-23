package es.albertosoto.spark_dtree.service;

import es.albertosoto.spark_dtree.util.SparkTestUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for SparkService
 * These tests use a real SparkSession and actual data files
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class SparkServiceIntegrationTest {

    private SparkSession sparkSession;
    private SparkService sparkService;
    private Path tempDataDir;

    @BeforeAll
    public void setup() throws IOException {
        // Create a test SparkSession
        sparkSession = SparkTestUtils.createTestSparkSession();
        
        // Create the SparkService with the test SparkSession
        sparkService = new SparkService(sparkSession);
        
        // Create a temporary directory for test data
        tempDataDir = Files.createTempDirectory("spark-test-data");
        
        // Create the resources directory structure
        Path resourcesDir = Paths.get(tempDataDir.toString(), "src", "main", "resources", "data");
        Files.createDirectories(resourcesDir);
        
        // Copy the test iris.csv file to the resources directory
        ClassPathResource testDataResource = new ClassPathResource("data/iris.csv");
        File sourceFile = testDataResource.getFile();
        Path targetPath = Paths.get(resourcesDir.toString(), "iris.csv");
        Files.copy(sourceFile.toPath(), targetPath, StandardCopyOption.REPLACE_EXISTING);
        
        // Set system property to point to the temp directory for file loading
        System.setProperty("user.dir", tempDataDir.toString());
    }
    
    @AfterAll
    public void tearDown() throws IOException {
        // Close the SparkSession
        if (sparkSession != null) {
            sparkSession.close();
        }
        
        // Clean up temporary files
        if (tempDataDir != null) {
            Files.walk(tempDataDir)
                .sorted((a, b) -> b.toString().length() - a.toString().length())
                .map(Path::toFile)
                .forEach(File::delete);
        }
    }
    
    @Test
    public void testTrainModelEndToEnd() {
        // Train the model
        Map<String, Object> result = sparkService.trainModel();
        
        // Verify training was successful
        assertNotNull(result);
        assertEquals("success", result.get("status"));
        assertTrue((double) result.get("accuracy") > 0);
        assertTrue((long) result.get("numTrainingSamples") > 0);
        assertTrue((long) result.get("numTestSamples") > 0);
        
        // Get model info
        Map<String, Object> modelInfo = sparkService.getModelInfo();
        
        // Verify model info
        assertNotNull(modelInfo);
        assertEquals("trained", modelInfo.get("status"));
        assertEquals("Model is trained and ready for predictions", modelInfo.get("message"));
        assertEquals("DecisionTreeClassifier", modelInfo.get("modelType"));
        
        // Make a prediction
        Map<String, Double> features = new HashMap<>();
        features.put("sepal_length", 5.1);
        features.put("sepal_width", 3.5);
        features.put("petal_length", 1.4);
        features.put("petal_width", 0.2);
        
        Map<String, Object> prediction = sparkService.predict(features);
        
        // Verify prediction
        assertNotNull(prediction);
        assertEquals("success", prediction.get("status"));
        assertNotNull(prediction.get("prediction"));
    }
}

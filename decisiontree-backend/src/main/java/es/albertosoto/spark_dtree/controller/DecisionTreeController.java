package es.albertosoto.spark_dtree.controller;

import es.albertosoto.spark_dtree.service.SparkService;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.springframework.beans.factory.annotation.Autowired;
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
    
    private final SparkService sparkService;
    
    @Autowired
    public DecisionTreeController(SparkService sparkService) {
        this.sparkService = sparkService;
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
        Map<String, Object> result = sparkService.trainModel();
        
        if ("error".equals(result.get("status"))) {
            return ResponseEntity.status(500).body(result);
        }
        
        return ResponseEntity.ok(result);
    }
    
    /**
     * Make a prediction using the trained model
     * @param features Map containing feature values
     * @return Prediction result
     */
    @PostMapping("/predict")
    public ResponseEntity<Map<String, Object>> predict(@RequestBody Map<String, Double> features) {
        Map<String, Object> result = sparkService.predict(features);
        
        if ("error".equals(result.get("status"))) {
            return ResponseEntity.status(400).body(result);
        }
        
        return ResponseEntity.ok(result);
    }
    
    /**
     * Get model information
     * @return Model information
     */
    @GetMapping("/model")
    public ResponseEntity<Map<String, Object>> getModelInfo() {
        Map<String, Object> info = sparkService.getModelInfo();
        return ResponseEntity.ok(info);
    }
}

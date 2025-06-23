package es.albertosoto.spark_dtree.controller;

import es.albertosoto.spark_dtree.service.SparkService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/**
 * Controller for Decision Tree operations
 */
@RestController
@RequestMapping("/api/dtree")
@Tag(name = "Decision Tree Controller", description = "API endpoints for Decision Tree operations with Apache Spark")
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
    @Operation(
        summary = "Get service information",
        description = "Returns basic information about the Decision Tree service"
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "200",
            description = "Successfully retrieved service information",
            content = @Content(mediaType = "application/json")
        )
    })
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
    @Operation(
        summary = "Train decision tree model",
        description = "Trains a decision tree model using the Iris dataset"
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "200",
            description = "Model trained successfully",
            content = @Content(mediaType = "application/json")
        ),
        @ApiResponse(
            responseCode = "500",
            description = "Error training model",
            content = @Content(mediaType = "application/json")
        )
    })
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
    @Operation(
        summary = "Make prediction",
        description = "Makes a prediction using the trained model and provided features"
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "200",
            description = "Prediction made successfully",
            content = @Content(mediaType = "application/json")
        ),
        @ApiResponse(
            responseCode = "400",
            description = "Invalid input or model not trained",
            content = @Content(mediaType = "application/json")
        )
    })
    @PostMapping("/predict")
    public ResponseEntity<Map<String, Object>> predict(
        @Parameter(description = "Feature values for prediction", required = true)
        @RequestBody Map<String, Double> features
    ) {
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
    @Operation(
        summary = "Get model information",
        description = "Returns information about the trained decision tree model"
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "200",
            description = "Model information retrieved successfully",
            content = @Content(mediaType = "application/json")
        )
    })
    @GetMapping("/model")
    public ResponseEntity<Map<String, Object>> getModelInfo() {
        Map<String, Object> info = sparkService.getModelInfo();
        return ResponseEntity.ok(info);
    }

    /**
     * Get the decision tree structure for frontend visualization
     * @return Tree structure in a format suitable for frontend rendering
     */
    @Operation(
        summary = "Get tree structure",
        description = "Returns the decision tree structure in a format suitable for frontend visualization"
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "200",
            description = "Tree structure retrieved successfully",
            content = @Content(mediaType = "application/json")
        ),
        @ApiResponse(
            responseCode = "404",
            description = "Model not trained yet",
            content = @Content(mediaType = "application/json")
        )
    })
    @GetMapping("/tree-structure")
    public ResponseEntity<Map<String, Object>> getTreeStructure() {
        Map<String, Object> result = sparkService.getTreeStructure();
        
        if ("not_trained".equals(result.get("status"))) {
            return ResponseEntity.status(404).body(result);
        } else if ("error".equals(result.get("status"))) {
            return ResponseEntity.status(500).body(result);
        }
        
        return ResponseEntity.ok(result);
    }
}

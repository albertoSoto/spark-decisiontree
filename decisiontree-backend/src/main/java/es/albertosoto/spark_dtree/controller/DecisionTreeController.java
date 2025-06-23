package es.albertosoto.spark_dtree.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.Map;

/**
 * Controller for Decision Tree operations
 */
@RestController
@RequestMapping("/api/dtree")
public class DecisionTreeController {

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
}

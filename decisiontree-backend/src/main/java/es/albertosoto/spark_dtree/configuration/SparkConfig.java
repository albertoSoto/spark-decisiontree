package es.albertosoto.spark_dtree.configuration;

import org.apache.spark.sql.SparkSession;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Created by Alberto Soto. 23/6/25
 */
@Configuration
public class SparkConfig {

    @Value("${spark.app.name}")
    private String appName;
    
    @Value("${spark.master}")
    private String masterUrl;

    @Bean
    public SparkSession sparkSession() {
        return SparkSession
                .builder()
                .appName(appName)
                .master(masterUrl)
                .getOrCreate();
    }
}

package es.albertosoto.spark_dtree.configuration;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.IOException;

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
    public SparkConf sparkConf() throws IOException {
        var conf = new SparkConf()
                .setAppName(appName)
                .setMaster(masterUrl);
        return conf;
    }

    @Bean
    public SparkSession sparkSession(JavaSparkContext sparkContext) {
        return SparkSession.builder()
                .sparkContext(sparkContext.sc())
                .config(sparkContext.getConf())
                .getOrCreate();
    }

    @Bean
    public JavaSparkContext javaSparkContext(SparkConf sparkConf) {
        return new JavaSparkContext(sparkConf);
    }
}

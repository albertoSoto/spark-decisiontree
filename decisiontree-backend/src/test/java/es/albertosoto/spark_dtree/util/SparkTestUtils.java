package es.albertosoto.spark_dtree.util;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;

/**
 * Utility class for Spark testing
 */
public class SparkTestUtils {

    /**
     * Create a test SparkSession for unit tests
     * @return SparkSession configured for testing
     */
    public static SparkSession createTestSparkSession() {
        return SparkSession.builder()
                .appName("SparkUnitTest")
                .master("local[1]")
                .config("spark.sql.shuffle.partitions", "1")
                .config("spark.ui.enabled", "false")
                .config("spark.driver.host", "localhost")
                .getOrCreate();
    }

    /**
     * Create a sample Iris dataset for testing
     * @param spark SparkSession
     * @return Dataset with sample Iris data
     */
    public static Dataset<Row> createSampleIrisDataset(SparkSession spark) {
        // Define the schema
        StructType schema = DataTypes.createStructType(new StructField[] {
            DataTypes.createStructField("sepal_length", DataTypes.DoubleType, false),
            DataTypes.createStructField("sepal_width", DataTypes.DoubleType, false),
            DataTypes.createStructField("petal_length", DataTypes.DoubleType, false),
            DataTypes.createStructField("petal_width", DataTypes.DoubleType, false),
            DataTypes.createStructField("species", DataTypes.StringType, false)
        });

        // Create sample data
        List<Row> rows = Arrays.asList(
            RowFactory.create(5.1, 3.5, 1.4, 0.2, "setosa"),
            RowFactory.create(4.9, 3.0, 1.4, 0.2, "setosa"),
            RowFactory.create(4.7, 3.2, 1.3, 0.2, "setosa"),
            RowFactory.create(7.0, 3.2, 4.7, 1.4, "versicolor"),
            RowFactory.create(6.4, 3.2, 4.5, 1.5, "versicolor"),
            RowFactory.create(6.9, 3.1, 4.9, 1.5, "versicolor"),
            RowFactory.create(6.3, 3.3, 6.0, 2.5, "virginica"),
            RowFactory.create(5.8, 2.7, 5.1, 1.9, "virginica"),
            RowFactory.create(7.1, 3.0, 5.9, 2.1, "virginica")
        );

        return spark.createDataFrame(rows, schema);
    }
}

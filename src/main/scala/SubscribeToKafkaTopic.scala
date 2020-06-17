import PublishToKafkaTopic.args
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.functions._

object SubscribeToKafkaTopic extends App {

  val kafkaTopic = args(0)
  val masterDataPath = args(1)
  val checkPointDir = args(2)

  val spark = SparkSession
    .builder
    .appName("Payments-Master-Data-Kafka")
    .master("local[*]")
    .getOrCreate()

  val mySchema = StructType(Array(
    StructField("step", IntegerType),
    StructField("type", StringType),
    StructField("amount", DoubleType),
    StructField("nameOrig", StringType),
    StructField("oldbalanceOrg", DoubleType),
    StructField("newbalanceOrig", DoubleType),
    StructField("nameDest", StringType),
    StructField("oldbalanceDest", DoubleType),
    StructField("newbalanceDest", DoubleType),
    StructField("isFraud", IntegerType),
    StructField("isFlaggedFraud", IntegerType)
  ))

  import spark.implicits._

  val kafkaDF = spark.
    readStream.
    format("kafka").
    option("kafka.bootstrap.servers", "localhost:9092").
    option("subscribe", kafkaTopic).
    option("startingOffsets", "earliest").
    load()

  val masterDataDF = kafkaDF.selectExpr("CAST(value AS STRING)").as[(String)].
    writeStream.
    outputMode("append").
    format("text").
    option("checkpointLocation", checkPointDir).
    start(masterDataPath)

  masterDataDF.awaitTermination()
}

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.functions._

object PublishToKafkaTopic extends App {

  val inputCSVPath = args(0)
  val kafkaTopic = args(1)
  val checkPointDir = args(2)

  println(inputCSVPath)
  println(kafkaTopic)
  println(checkPointDir)

  val spark = SparkSession
    .builder
    .appName("Publish-Payments-Kafka")
    .master("local")
    .getOrCreate()
  import spark.implicits._

  //Schema to enforce on csv master data
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

  val streamingDataFrame = spark.readStream.option("header", "true").schema(mySchema).csv(inputCSVPath)

  val keyValueDF = streamingDataFrame.
    selectExpr("CAST(step AS STRING) AS key", "to_json(struct(*)) AS value")

  val query = keyValueDF.
    writeStream
    .format("kafka")
    .option("topic", kafkaTopic)
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("checkpointLocation", checkPointDir)
    .start()

  query.awaitTermination()

}

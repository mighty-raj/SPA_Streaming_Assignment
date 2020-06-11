import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.functions._

object PublishToKafkaTopic extends App {

  val inputCSVPath = args(0)
  println(inputCSVPath)

  val kafkaTopic = args(1)
  println(kafkaTopic)

  val checkPointDir = args(2)
  println(checkPointDir)

  val spark = SparkSession
    .builder
    .appName("Publish-Payments-Kafka")
    .master("local")
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

  val streamingDataFrame = spark.readStream.option("header", "true").schema(mySchema).csv(inputCSVPath)
  val resDf = streamingDataFrame.select(
    'step.cast("String").as("key"),
    concat_ws(",",streamingDataFrame.columns.map(c => col(c)): _*).as("value")
  )

  val query = resDf.
    writeStream
    .format("kafka")
    .option("topic", kafkaTopic)
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("checkpointLocation", checkPointDir)
    .start()


  /*val query = streamingDataFrame.
    //    selectExpr("CAST(step AS STRING) AS key", "to_json(struct(*)) AS value").
    writeStream
    .format("console")
    .start()*/

  query.awaitTermination()

}

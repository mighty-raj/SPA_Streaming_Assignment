import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.log4j.{Level, Logger}

object PredictRealTime extends App {

  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)

  val kafkaTopic = args(0)
  val logRegModelPath = args(1)

  val spark = SparkSession
    .builder
    .appName("Real-Time-Fraud-Txn-Dectector")
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._

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

  val kafkaDF = spark.
    readStream.
    format("kafka").
    option("kafka.bootstrap.servers", "localhost:9092").
    option("subscribe", kafkaTopic).
    option("startingOffsets", "earliest").
    load()

  println("Kafka Streaming source created ...")

  val paySimDF = kafkaDF.
    select(from_json(col("value").cast("string"), mySchema).alias("paysim_txn")).
    select("paysim_txn.*").
    select(
      'step,
      'amount,
      'oldbalanceOrg,
      'newbalanceOrig,
      'oldbalanceDest,
      'newbalanceDest
    )
  println("New PaySim Transaction Formatted and Ready to Consume ...")
  paySimDF.printSchema()

  //Load model from disk
  val logisticRegressionModelLoaded = PipelineModel.load(logRegModelPath)
  println("Loaded ML Model From Disk ... ")

  // prediction of pass/fail status of sample data set
  val paySimPredictedDf = logisticRegressionModelLoaded.
    transform(paySimDF).
    select(
      'step,
      'amount,
      'oldbalanceOrg,
      'newbalanceOrig,
      'oldbalanceDest,
      'newbalanceDest,
      'prediction
    )

  paySimPredictedDf.printSchema()

  /*paySimPredictedDf.
    filter("prediction == 1.0d").
    writeStream.
    format("console").
    option("truncate","false").
    start().
    awaitTermination()*/

  //save data to different kafka topics based on the predicted value!!
  paySimPredictedDf.
    writeStream.foreachBatch { (batchDF: DataFrame, batchId: Long) =>
    batchDF.persist() // Caching Dataframe

    //Predicted as FRAUD Transactions!
    batchDF.
      filter("prediction == 1.0d"). // 1.0d => FRAUD Txn
      selectExpr("CAST(step AS STRING) AS key", "to_json(struct(*)) AS value").
      write.format("kafka").
      option("kafka.bootstrap.servers", "localhost:9092").
      option("topic", "FraudTxns").
      save()

    //Predicted as NORMAL Transactions!
    batchDF.
      filter("prediction == 0.0d"). // 0.0d => Normal Txn
      selectExpr("CAST(step AS STRING) AS key", "to_json(struct(*)) AS value").
      write.
      format("kafka").
      option("kafka.bootstrap.servers", "localhost:9092").
      option("topic", "NormalTxns").
      save()

    batchDF.unpersist() //Un-Caching Dataframe to free-up memory
  }.
    start().
    awaitTermination()
}

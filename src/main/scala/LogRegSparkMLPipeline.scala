import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._

object LogRegSparkMLPipeline extends App {
  val inpCsv = args(0)
  val modelSavePath = args(1)
  val kafkaValidationDataPath = args(2)

  // context for spark
  val spark = SparkSession.builder
    .master("local[*]")
    .appName("PaySim-LogisticReg")
    .getOrCreate()

  // SparkSession has implicits
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

  //Read Training Data into Spark DataFrame
  val paysimDF = spark.read.format("csv")
    .option("header", value = true)
    .option("delimiter", ",")
    .option("mode", "DROPMALFORMED")
    .schema(mySchema)
    .load(inpCsv)
    .cache()
    .filter($"type" === "TRANSFER" || $"type" === "CASH_OUT")

  val typeIndexer = new StringIndexer()
    .setInputCol("type")
    .setOutputCol("typeIndex")

  val typeIndexedDF = typeIndexer.fit(paysimDF).transform(paysimDF)


  // columns that need to added to feature column
  val cols = Array("typeIndex", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")

  // VectorAssembler to add feature column || input columns - cols || feature column - features
  val assembler = new VectorAssembler()
    .setInputCols(cols)
    .setOutputCol("features")

  val featureDf = assembler.transform(typeIndexedDF)
  featureDf.printSchema()
  featureDf.show(10)

  // StringIndexer define new 'label' column with 'result' column
  val indexer = new StringIndexer()
    .setInputCol("isFraud")
    .setOutputCol("label")

  // split data set training and test || training data set - 70% || test data set - 30%
  val seed = 5043

  // we run paysimDF on the pipeline, so split paysimDF
  val Array(pipelineTrainingData, pipelineTestingData) = typeIndexedDF.randomSplit(Array(0.7, 0.3), seed)

  // train logistic regression model with training data set
  val logisticRegression = new LogisticRegression()
    .setMaxIter(500)
    .setRegParam(0.02)
    .setElasticNetParam(0.8)

  // VectorAssembler and StringIndexer are transformers
  // LogisticRegression is the estimator
  val stages = Array(assembler, indexer, logisticRegression)

  // build pipeline
  val pipeline = new Pipeline().setStages(stages)

  val pipelineModel = pipeline.fit(pipelineTrainingData)

  println("pipelineTestingData SCHEMA Below ===>>> ")
  pipelineTestingData.printSchema()

  //Random Split again to just save some test data for validation thru kafka
  val Array(pipelineValidationData, kafkaValidationData) = pipelineTestingData.randomSplit(Array(0.7,0.3), seed)

  //Save Kafka Validation Data as csv
  kafkaValidationData.coalesce(1)
    .write
    .option("sep",",")
    .mode("overwrite")
    .csv(kafkaValidationDataPath)

  // test model with test data
  val pipelinePredictionDf = pipelineModel.transform(pipelineValidationData)
  pipelinePredictionDf.filter("prediction == 1.0d").show(200, false)

  // evaluate model with area under ROC
  val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("prediction")
    .setMetricName("areaUnderROC")

  // measure the accuracy of pipeline model
  val pipelineAccuracy = evaluator.evaluate(pipelinePredictionDf)
  println(pipelineAccuracy)

  // save model
  pipelineModel.write.overwrite()
    .save(modelSavePath)
}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._

object SparkMLLogisticReg extends App{

  val inpCsv = args(0)

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

  // read to DataFrame
  val paysimDF = spark.read.format("csv")
    .option("header", value = true)
    .option("delimiter", ",")
    .option("mode", "DROPMALFORMED")
    .schema(mySchema)
//    .load(getClass.getResource(inpCsv).getPath)
    .load(inpCsv)
    .cache()

  // columns that need to added to feature column
  val cols = Array("step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")

  // VectorAssembler to add feature column
  // input columns - cols
  // feature column - features
  val assembler = new VectorAssembler()
    .setInputCols(cols)
    .setOutputCol("features")
  val featureDf = assembler.transform(paysimDF)
  featureDf.printSchema()
  featureDf.show(10)

  // StringIndexer define new 'label' column with 'result' column
  val indexer = new StringIndexer()
    .setInputCol("isFraud")
    .setOutputCol("label")

  val labelDf = indexer.fit(featureDf).transform(featureDf)
  labelDf.printSchema()
  labelDf.show(10)

  // split data set training and test
  // training data set - 70%
  // test data set - 30%
  val seed = 5043
  val Array(trainingData, testData) = labelDf.randomSplit(Array(0.7, 0.3), seed)

  // train logistic regression model with training data set
  val logisticRegression = new LogisticRegression()
    .setMaxIter(100)
    .setRegParam(0.02)
    .setElasticNetParam(0.8)

  val logisticRegressionModel = logisticRegression.fit(trainingData)

  // run model with test data set to get predictions
  // this will add new columns rawPrediction, probability and prediction
  val predictionDf = logisticRegressionModel.transform(testData)
  predictionDf.show(10)

  // evaluate model with area under ROC
  val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("prediction")
    .setMetricName("areaUnderROC")

  // measure the accuracy
  val accuracy = evaluator.evaluate(predictionDf)
  println(accuracy)


}

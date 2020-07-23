import com.salesforce.op.features.FeatureBuilder
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.types._
import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.regression.RegressionModelSelector
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelSelector
import com.salesforce.op.stages.impl.classification._
import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.readers.DataReaders
import com.salesforce.op.stages.impl.classification.MultiClassificationModelSelector
import java.util.Calendar
import java.io._

object AzureBlobAnalysisv2 {
  def main(args: Array[String]) {
    LogManager.getLogger("com.salesforce.op").setLevel(Level.ERROR)
    val conf = new SparkConf()
    conf.setAppName("AutoMLForAll")

    implicit val spark = SparkSession.builder.config(conf).getOrCreate()
    val confh=new org.apache.hadoop.conf.Configuration()

    val targetColumn = "class" 

    for(fold <- 1 to 10){
      println("Fold: " + fold); 

      var fold_folder = "/home/lferreira/autoautoml/new_data/credit/transmogrifai/fold" + fold.toString()
      
      var train_df = spark.sqlContext.read.format("csv").option("header", "true").option("inferSchema", "true").load(fold_folder + "/train.csv")

      var toBechanged = train_df.schema.fields.filter(x => x.dataType == IntegerType || x.dataType == LongType)
      toBechanged.foreach({ row =>
      train_df = train_df.withColumn(row.name.concat("tmp"), train_df.col(row.name).cast(DoubleType))
        .drop(row.name)
        .withColumnRenamed(row.name.concat("tmp"), row.name)
      })

      val (saleprice, features) = FeatureBuilder.fromDataFrame[RealNN](train_df, response = targetColumn)
      val featureVector = features.toSeq.autoTransform()
      val checkedFeatures = saleprice.sanityCheck(featureVector, checkSample = 1.0, removeBadFeatures = true)
      val pred = BinaryClassificationModelSelector.withCrossValidation(numFolds = 5, validationMetric = Evaluators.BinaryClassification.auROC).setInput(saleprice, checkedFeatures).getOutput()
      val wf = new OpWorkflow()

      var start = Calendar.getInstance.getTime
      var model = wf.setInputDataset(train_df).setResultFeatures(pred).train()
      var end = Calendar.getInstance.getTime

      print(model.summaryPretty())

      var summary = model.summaryPretty()

      val evaluator = Evaluators.BinaryClassification().setLabelCol(saleprice).setPredictionCol(pred)

      var testData = spark.sqlContext.read.format("csv").option("header", "true").option("inferSchema", "true").load(fold_folder + "/test.csv")
      var toBechanged2 = testData.schema.fields.filter(x => x.dataType == IntegerType || x.dataType == LongType)
      toBechanged2.foreach({ row =>
        testData = testData.withColumn(row.name.concat("tmp"), testData.col(row.name).cast(DoubleType))
          .drop(row.name)
          .withColumnRenamed(row.name.concat("tmp"), row.name)
      })
      
      var preds = model.setInputDataset(testData).scoreAndEvaluate(evaluator).toString()

      var w = new BufferedWriter(new FileWriter(fold_folder + "/perf.txt"))
      w.write("START\n")
      w.write(start.toString())
      w.write("\n\n\n\nEND\n")
      w.write(end.toString())
      w.write("\n\n\n\nPREDS\n")
      w.write(preds)
      w.write("\n\n\n\nSUMMARY\n")
      w.write(summary)
      w.close()

      model.save(fold_folder + "/model")

    }

    // Read data as a DataFrame
    var passengersData = spark.sqlContext.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("/home/lferreira/autoautoml/data/cholesterol/cholesterol-train.csv")

    var passengersData = spark.sqlContext.read.format("csv").option("header", "true").option("inferSchema", "true").load("/home/lferreira/autoautoml/data/mfeat/mfeat-train.csv")


    val passengersData = DataReaders.Simple.csvCase[Liver](Option("/home/lferreira/autoautoml/data/liver-disorders/liver-disorders-train.csv")).readDataset().toDF()

    
    val targetColumn = spark.sparkContext.wholeTextFiles("drinks").take(1)(0)._2
    val targetColumn = "class"
    //Convert Int and Long to Double to avoid Feature Builder exception with Integer / Long Types
    val toBechanged = passengersData.schema.fields.filter(x => x.dataType == IntegerType || x.dataType == LongType)
    toBechanged.foreach({ row =>
      passengersData = passengersData.withColumn(row.name.concat("tmp"), passengersData.col(row.name).cast(DoubleType))
        .drop(row.name)
        .withColumnRenamed(row.name.concat("tmp"), row.name)
    })
    //Let's try to understand from the target variable which ML problem we want to solve
    val view = passengersData.createOrReplaceTempView("myview")
    val countTarget = spark.sql("SELECT COUNT(DISTINCT " + targetColumn + ") FROM myview").take(1)(0).get(0).toString().toInt
    val targetType = passengersData.schema.fields.filter(x => x.name == targetColumn).take(1)(0).dataType
    //Max Distinct Values for Binary Classification is 2 and for multi class is 30
    val binaryL: Int = 2
    val multiL: Int = 30

    //If the target variable has 2 distinct values and it is numeric can be a binary classification
    if (countTarget == binaryL && targetType == DoubleType) {
      val (saleprice, features) = FeatureBuilder.fromDataFrame[RealNN](passengersData, response = targetColumn)
      val featureVector = features.toSeq.autoTransform()
      val checkedFeatures = saleprice.sanityCheck(featureVector, checkSample = 1.0, removeBadFeatures = true)
      val pred = BinaryClassificationModelSelector.withCrossValidation(numFolds = 5, validationMetric = Evaluators.BinaryClassification.auROC).setInput(saleprice, checkedFeatures).getOutput()
      val wf = new OpWorkflow()

      val start = Calendar.getInstance.getTime
      val model = wf.setInputDataset(passengersData).setResultFeatures(pred).train()
      val end = Calendar.getInstance.getTime
      
      print(model.summaryPretty())

      val evaluator = Evaluators.BinaryClassification().setLabelCol(saleprice).setPredictionCol(pred)

      model.setInputDataset(testData).scoreAndEvaluate(evaluator)

      model.save("/home/lferreira/autoautoml/data/churn/transmogrifai")

      val results = "Model summary:\n" + model.summaryPretty()
      model.save("wasbs://REPLACETHIS@REPLACETHIS.blob.core.windows.net/models/" + uniqueId + "/binmodel")
      val dfWrite = spark.sparkContext.parallelize(Seq(results))
      dfWrite.coalesce(1).saveAsTextFile("wasbs://REPLACETHIS@REPLACETHIS.blob.core.windows.net/results/" + uniqueId + ".txt")
    }
    //If the target variable has more that 2 distinct values , less than 30 and it is string type can be a multi-classification
    else if (countTarget > binaryL && countTarget < multiL && targetType == StringType) {
      val (saleprice, features) = FeatureBuilder.fromDataFrame[RealNN](passengersData, response = targetColumn)
      val featureVector = features.toSeq.autoTransform()
      val checkedFeatures = saleprice.sanityCheck(featureVector, checkSample = 1.0, removeBadFeatures = true)
      val pred = MultiClassificationModelSelector.withCrossValidation(numFolds = 5, validationMetric = Evaluators.MultiClassification.f1).setInput(saleprice, checkedFeatures).getOutput()
      
      val wf = new OpWorkflow()
      

      val start = Calendar.getInstance.getTime
      val model = wf.setInputDataset(passengersData).setResultFeatures(pred).train()
      val end = Calendar.getInstance.getTime

      print(model.summaryPretty())

      val evaluator = Evaluators.MultiClassification().setLabelCol(saleprice).setPredictionCol(pred)
      
      val results = "Model summary:\n" + model.summaryPretty()
      model.save("wasbs://REPLACETHIS@REPLACETHIS.blob.core.windows.net/models/" + uniqueId + "/multicmodel")
      val dfWrite = spark.sparkContext.parallelize(Seq(results))
      dfWrite.coalesce(1).saveAsTextFile("wasbs://REPLACETHIS@REPLACETHIS.blob.core.windows.net/results/" + uniqueId + ".txt")
    }
    // If it's not a classification then we can try a regression
    else {
      val (saleprice, features) = FeatureBuilder.fromDataFrame[RealNN](passengersData, response = targetColumn)
      val featureVector = features.toSeq.autoTransform()
      val checkedFeatures = saleprice.sanityCheck(featureVector, checkSample = 1.0, removeBadFeatures = true)
      val pred = RegressionModelSelector.withCrossValidation(numFolds = 5, validationMetric = Evaluators.Regression.mae).setInput(saleprice, checkedFeatures).getOutput()
      val wf = new OpWorkflow()

      val start = Calendar.getInstance.getTime
      val model = wf.setInputDataset(passengersData).setResultFeatures(pred).train()
      val end = Calendar.getInstance.getTime

      print(model.summaryPretty())

      val evaluator = Evaluators.Regression().setLabelCol(saleprice).setPredictionCol(pred)

      
      model.setInputDataset(testData).scoreAndEvaluate(evaluator)

      val results = "Model summary:\n" + model.summaryPretty()
      model.save("/home/lferreira/autoautoml/data/plasma/transmogrfai")
      val dfWrite = spark.sparkContext.parallelize(Seq(results))

    }

    var testData = spark.sqlContext.read.format("csv").option("header", "true").option("inferSchema", "true").load("/home/lferreira/autoautoml/data/mfeat/mfeat-test.csv")
    val toBechanged = testData.schema.fields.filter(x => x.dataType == IntegerType || x.dataType == LongType)
    toBechanged.foreach({ row =>
      testData = testData.withColumn(row.name.concat("tmp"), testData.col(row.name).cast(DoubleType))
        .drop(row.name)
        .withColumnRenamed(row.name.concat("tmp"), row.name)
    })
    model.setInputDataset(testData).scoreAndEvaluate(evaluator)
    
    spark.close()
  }
}
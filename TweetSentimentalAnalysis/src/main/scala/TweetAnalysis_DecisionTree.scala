



  import org.apache.spark.ml.feature.RegexTokenizer
  import org.apache.spark.sql.{Row, SparkSession}
  import org.apache.spark.ml.feature.StopWordsRemover
  import org.apache.spark.ml.feature.StringIndexer
  import org.apache.spark.ml.Pipeline
  import org.apache.spark.ml.feature.HashingTF

  import scala.collection.mutable.ListBuffer
  import org.apache.spark.ml.classification.DecisionTreeClassificationModel
  import org.apache.spark.ml.classification.DecisionTreeClassifier
  import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
  import org.apache.spark.ml.feature.{IndexToString, VectorIndexer}
  import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
  import org.apache.spark.mllib.evaluation.MulticlassMetrics





  object TweetAnalysis_DecisionTree {
    def main(args: Array[String]): Unit = {

      if (args.length == 0) {
        println("i need  two parameters ")
      }
      if(args.length < 2) {
        println("i need two arguments")
      }

      val spark = SparkSession
        .builder()
        .appName("Analysis")
        .getOrCreate()



      val sc = spark.sparkContext

      import spark.implicits._





      val ratings = spark.read.option("header", "true")
        .option("inferSchema", "true")
        .option("delimiter", ",")
        .csv(args(0));

      val ratings1 = ratings.filter(ratings.col("text").isNotNull);
      val indexModel = new StringIndexer()
        .setInputCol("airline_sentiment")
        .setOutputCol("label").fit(ratings1)
      val regexTokenizer = new RegexTokenizer()
        .setInputCol("text")
        .setOutputCol("words")
        .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

      val remover = new StopWordsRemover().setInputCol(regexTokenizer.getOutputCol).setOutputCol("filtered")



      val hashingTF = new HashingTF()
        .setInputCol(remover.getOutputCol)
        .setOutputCol("features")




      // Split the data into training and test sets (30% held out for testing).
      val Array(trainingData, testData) = ratings1.randomSplit(Array(0.7, 0.3))

      // Train a DecisionTree model.
      val dt = new DecisionTreeClassifier()
        .setLabelCol("label")
        .setFeaturesCol("features")



      // Chain indexers and tree in a Pipeline.
      val pipeline = new Pipeline()
        .setStages(Array( indexModel,regexTokenizer, remover, hashingTF, dt))
      val paramGrid = new ParamGridBuilder()
        .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
        .build()

      // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
      // This will allow us to jointly choose parameters for all Pipeline stages.
      // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
      // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
      // is areaUnderROC.
      val cv = new CrossValidator()
        .setEstimator(pipeline)
        .setEvaluator(new MulticlassClassificationEvaluator)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(2)  // Use 3+ in practice
        .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel

      // Run cross-validation, and choose the best set of parameters.



      val cvModel = cv.fit(trainingData)

      val predictionAndLabels = cvModel.transform(testData).select("prediction", "label").map{case Row( prediction: Double, label: Double) =>
        println(s"(prediction=$prediction")
        (prediction, label)


      }



      val metrics = new MulticlassMetrics(predictionAndLabels.rdd)


      // Confusion matrix
      println("Confusion matrix:")
      println(metrics.confusionMatrix)



      var fruits = new ListBuffer[String]()
      fruits += s"Confusion Matrix: ${metrics.confusionMatrix}"
      // Overall Statistics
      val accuracy = metrics.accuracy




      println("Summary Statistics")
      println(s"Accuracy = $accuracy")

      fruits += s"Accuracy = $accuracy"
      // Precision by label
      val labels = metrics.labels
      labels.foreach { l =>
        println(s"Precision($l) = " + metrics.precision(l))
        fruits += s"Precision($l) : ${metrics.precision(l)}"
      }

      // Recall by label
      labels.foreach { l =>
        println(s"Recall($l) = " + metrics.recall(l))
        fruits += s"Recall($l) : ${metrics.recall(l)}"
      }

      // False positive rate by label
      labels.foreach { l =>
        println(s"FPR($l) = " + metrics.falsePositiveRate(l))
        fruits += s"FPR($l): ${metrics.falsePositiveRate(l)}"
      }

      // F-measure by label
      labels.foreach { l =>
        println(s"F1-Score($l) = " + metrics.fMeasure(l))
        fruits += s"F1-Score($l): ${metrics.fMeasure(l)}"
      }

      // Weighted stats
      println(s"Weighted precision: ${metrics.weightedPrecision}")
      fruits += s"Weighted precision: ${metrics.weightedPrecision}"
      println(s"Weighted recall: ${metrics.weightedRecall}")
      fruits += s"Weighted recall: ${metrics.weightedRecall}"
      println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
      fruits += s"Weighted F1 score: ${metrics.weightedFMeasure}"
      println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")
      fruits += s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}"



      val ListOfLabels = fruits.toList

      val rdd1 = sc.parallelize(ListOfLabels)
      rdd1.saveAsTextFile(args(1))
      sc.stop()



    }

}

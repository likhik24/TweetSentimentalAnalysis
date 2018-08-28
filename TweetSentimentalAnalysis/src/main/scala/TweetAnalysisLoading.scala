

/************************************************************
    * This class requires two arguments:
    *  input file - can be downloaded from http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip
    *  output location - can be on S3 or cluster
    *************************************************************/



import org.apache.spark.ml.feature.{RegexTokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row
import scala.collection.mutable.ListBuffer

import org.apache.spark.mllib.evaluation.MulticlassMetrics




  object TweetAnalysisLoading {
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
      val lr = new LogisticRegression()
        .setMaxIter(10)
      val pipeline = new Pipeline()
        .setStages(Array(indexModel,regexTokenizer, remover, hashingTF, lr))

      // We use a ParamGridBuilder to construct a grid of parameters to search over.
      // With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
      // this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
      val paramGrid = new ParamGridBuilder()
        .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
        .addGrid(lr.regParam, Array(0.1, 0.01))
        .build()

      // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
      // This will allow us to jointly choose parameters for all Pipeline stages.
      // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
      // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
      // is areaUnderROC.
      val cv = new CrossValidator()
        .setEstimator(pipeline)
        .setEvaluator(new BinaryClassificationEvaluator)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(2)  // Use 3+ in practice
        .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel

      // Run cross-validation, and choose the best set of parameters.

      val Array(training,test) = ratings1.randomSplit(Array(0.8,0.2))

      val cvModel = cv.fit(training)

      val predictionAndLabels = cvModel.transform(test).select("prediction", "label").map{case Row( prediction: Double, label: Double) =>
        println(s"(prediction=$prediction")
        (prediction, label)


      }



      // Instantiate metrics object
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

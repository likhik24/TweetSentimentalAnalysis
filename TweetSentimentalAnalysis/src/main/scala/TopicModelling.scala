import org.apache.spark.ml.feature.{RegexTokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.functions._
import scala.collection.mutable
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row
import scala.collection.mutable.ListBuffer

import org.apache.spark.mllib.evaluation.MulticlassMetrics




object TopicModelling {
  def main(args: Array[String]): Unit = {

    if (args.length == 0) {
      println("i need  two parameters ")
    }
    if (args.length < 2) {
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

    val updatedDf = ratings1.withColumn("airline_sentiment", regexp_replace(col("airline_sentiment"), "positive", "5.0"))
    val updatedDf1 = updatedDf.withColumn("airline_sentiment", regexp_replace(col("airline_sentiment"), "neutral", "2.5"))
    val updatedDf2 = updatedDf1.withColumn("airline_sentiment", regexp_replace(col("airline_sentiment"), "negative", "1.0"))
    updatedDf2.withColumn("airline_sentiment", when($"airline_sentiment" === "5.0", 5.0))
    updatedDf2.withColumn("airline_sentiment", when($"airline_sentiment" === "1.0", 1.0))
    updatedDf2.withColumn("airline_sentiment", when($"airline_sentiment" === "2.5", 2.5))
    updatedDf2.select("airline_sentiment").show()
    val df = updatedDf2.selectExpr( "tweet_id","cast(airline_sentiment as double) airline_sentiment","airline_sentiment_confidence","negativereason","negativereason_confidence","airline","airline_sentiment_gold","name","negativereason_gold","retweet_count","text","tweet_coord","tweet_created","tweet_location","user_timezone")
    val air_ratings = df.groupBy("airline").avg("airline_sentiment").toDF("airline","airline_sentiment")
    //calculate the average rating for each airline, and then figure out which are the best in terms of average ratings
    val goodairline = air_ratings.orderBy(desc("airline_sentiment")).limit(1)
    //calculate the average rating for each airline, and then figure out which are the worst in terms of average ratings
    val badairline = air_ratings.orderBy("airline_sentiment").limit(1)
    val k = goodairline.select("airline").as[String].collect()
    val k1 = badairline.select("airline").as[String].collect()
    val data = df.filter((df.col("airline").contains(k(0))) )
    val data1 = df.filter((df.col("airline").contains(k1(0))))


    val termsdataframe = data.select("text").map(row => row.mkString(","))
    val rdd1 = termsdataframe.rdd
    val termsdataframe1 = data1.select("text").map(row => row.mkString(","))

    val rdd3 = termsdataframe1.rdd
    //removing stopwords
    val testWords = Set("tis","'twas","a","able","about","across","after","ain't","all","almost","also","am","among","an","and","any","are","aren't","as","at","be","because","been","but","by","can","can't","cannot","could","could've","couldn't","dear","did","didn't","do","does","doesn't","don't","either","else","ever","every","for","from","get","got","had","has","hasn't","have","he","he'd","he'll","he's","her","hers","him","his","how","how'd","how'll","how's","however","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","just","least","let","like","likely","may","me","might","might've","mightn't","most","must","must've","mustn't","my","neither","no","nor","not","of","off","often","on","only","or","other","our","own","rather","said","say","says","shan't","she","she'd","she'll","she's","should","should've","shouldn't","since","so","some","than","that","that'll","that's","the","their","them","then","there","there's","these","they","they'd","they'll","they're","they've","this","tis","to","too","twas","us","wants","was","wasn't","we","we'd","we'll","we're","were","weren't","what","what'd","what's","when","when","when'd","when'll","when's","where","where'd","where'll","where's","which","while","who","who'd","who'll","who's","whom","why","why'd","why'll","why's","will","with","won't","would","would've","wouldn't","yet","you","you'd","you'll","you're","you've","your")

    var fruits = new ListBuffer[String]()
    val tokenized =
      rdd1.map(_.toLowerCase.split("\\s")).map(_.filter(_.length > 3)
        .filter(_.forall(java.lang.Character.isLetter)).filter(!testWords.contains(_)))
    val termCounts: Array[(String, Long)] =
      tokenized.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)

    val vocabArray: Array[String] =
      termCounts.takeRight(termCounts.size).map(_._1)

    val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap
    val documents =
      tokenized.zipWithIndex.map { case (tokens, id) =>
        val counts = new mutable.HashMap[Int, Double]()
        tokens.foreach { term =>
          if (vocab.contains(term)) {
            val idx = vocab(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(vocab.size, counts.toSeq))
      }
    val numTopics = 10
    val lda = new LDA().setK(numTopics).setMaxIterations(10)
    val ldaModel = lda.run(documents)
    fruits += "the good airlines topics are"
    // Print topics, showing top-weighted 10 terms for each topic.
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
    topicIndices.foreach { case (terms, termWeights) =>
      println("TOPIC:")
      fruits += "TOPIC:"
      terms.zip(termWeights).foreach { case (term, weight) =>
        println(s"${vocabArray(term.toInt)}\t$weight")
        fruits += s"${vocabArray(term.toInt)}\t$weight"
      }
      println()
      fruits += " "
    }

    val tokenized1 =
      rdd3.map(_.toLowerCase.split("\\s")).map(_.filter(_.length > 3)
        .filter(_.forall(java.lang.Character.isLetter)).filter(!testWords.contains(_)))
    val termCounts1: Array[(String, Long)] =
      tokenized1.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)

    val vocabArray1: Array[String] =
      termCounts1.takeRight(termCounts1.size).map(_._1)


    val vocab1: Map[String, Int] = vocabArray1.zipWithIndex.toMap
    val documents1 =
      tokenized1.zipWithIndex.map { case (tokens, id) =>
        val counts = new mutable.HashMap[Int, Double]()
        tokens.foreach { term =>
          if (vocab1.contains(term)) {
            val idx = vocab1(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(vocab1.size, counts.toSeq))
      }
    val numTopics1 = 10
    val lda1 = new LDA().setK(numTopics1).setMaxIterations(10)

    val ldaModel1 = lda1.run(documents1)
    fruits += "the bad airlines topics are"
    // Print topics, showing top-weighted 10 terms for each topic.
    val topicIndices1 = ldaModel1.describeTopics(maxTermsPerTopic = 10)
    topicIndices1.foreach { case (terms, termWeights) =>
      println("TOPIC:")
      fruits += "TOPIC:"
      terms.zip(termWeights).foreach { case (term, weight) =>
        println(s"${vocabArray1(term.toInt)}\t$weight")
        fruits += s"${vocabArray1(term.toInt)}\t$weight"
      }
      println()
      fruits += " "
    }


    println(fruits.toList)
    val ListOfLabels = fruits.toList
    val rdd2 = sc.parallelize(ListOfLabels)
    rdd2.saveAsTextFile(args(1))
    sc.stop()







  }
}
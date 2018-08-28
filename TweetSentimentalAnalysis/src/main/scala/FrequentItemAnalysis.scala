
import org.apache.spark.sql.SparkSession

import org.apache.spark.sql.functions._
import org.apache.spark.ml.fpm.FPGrowth

import scala.collection.mutable.ListBuffer


object FrequentItemAnalysis {

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


    val df_instacart_products = spark.read.option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ",")
      .csv(args(0));

    val trans = df_instacart_products.groupBy("order_id").agg(collect_list("product_id") as "product")
    val fpgrowth = new FPGrowth().setItemsCol("product").setMinSupport(0.005).setMinConfidence(0.3)

    val model = fpgrowth.fit(trans)

    val frequency = model.freqItemsets.orderBy(desc("freq")).limit(10).take(10)

    val association = model.associationRules.orderBy(desc("confidence")).limit(10).take(10)

    val frequencyRDD = sc.parallelize(frequency)

    val associationRDD = sc.parallelize(association)

    var fruits = new ListBuffer[String]()

    fruits += "The top 10 frequent item sets are" + "\n" + "\n" + "product_id        order_id"

    val rdd1 = sc.parallelize(fruits)
    var fruits1 = new ListBuffer[String]()

    fruits1 += "The top 10 association rules are" + "\n"

    val rdd2 = sc.parallelize(fruits1)

    val result = rdd1 ++ frequencyRDD.map(_.mkString(",")) ++ rdd2 ++ associationRDD.map(_.mkString(","))

    result.saveAsTextFile(args(1))
  }
}

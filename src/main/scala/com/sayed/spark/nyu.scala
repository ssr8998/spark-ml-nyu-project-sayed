package com.sayed.spark

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{Imputer, StandardScaler, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, mean, stddev}

object nyu {
  def main(args:Array[String]){

    val spark: SparkSession = SparkSession.builder()
      .master("local")
      .appName("NYU")
      .getOrCreate()



    //-------------------- CONSTANTS -----------------------------------------
    val inputFile=args(0)
    val tempFile=args(1)

    // val inputFile = "hdfs:///user/ss13337/project/data0-4.csv"
    //val tempFile = "/user/ss13337/project/temp/pca_op"

    //-------------------- DATA LOADING  -------------------------------------

    var data = spark.read.option("header","true").option("inferSchema","true").format("csv").load(inputFile)

    //---------------------REMOVE OUTLIERS--------------------------------

    val outlier_strength:Int = 3
    var cols = data.columns.takeRight(1)
    for (c <- cols) {
      val stats = data.agg(mean(c).as("mean"), stddev(c).as("stddev")).withColumn("UpperLimit", col("mean") + col("stddev") * outlier_strength)
      data = data.filter(data(c) < stats.first().get(2))
    }

    //-------------------- HANDLE NaN VALUES  -----------------------

    //Replaces the NaN values with mean of that column
    cols = data.columns.dropRight(1)
    val imputedModel = new Imputer().setInputCols(cols).setOutputCols(cols).fit(data)
    data = imputedModel.transform(data)


    //---------------------INTO LIBSVM FORMAT---------------------------------

    // Transform data into Label and Vector of Features for PCA
    import spark.implicits._
    val vec = data.map{ row =>
      new LabeledPoint(row.getDouble(4), Vectors.dense(row.getDouble(0), row.getDouble(1), row.getDouble(2), row.getDouble(3)))}.rdd.cache()

    //----------------------TRANSFORM DATA---------------------------------

    // Transform data into Label and Vector of Features
    val vectorAssembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
    data = vectorAssembler.transform(data)


    //--------------------- LR MODEL PERFORMANCE WITHOUT PCA -----------------------------

    // Scaling
    val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)
    val scalerModel = scaler.fit(data)
    data = scalerModel.transform(data)

    // ML splits
    var Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2))

    // LinearRegression
    val lir = new LinearRegression().setLabelCol("4").setFeaturesCol("scaledFeatures").setMaxIter(100).setRegParam(0.1).setElasticNetParam(1.0)
    val model = lir.fit(trainingData)
    val pred = model.evaluate(testData)

    // Evaluation
    val evaluator = new RegressionEvaluator().setLabelCol("4").setPredictionCol("prediction").setMetricName("rmse")
    val rmse1 = evaluator.evaluate(pred.predictions)


    //--------------------- PCA PROCEDURE -------------------------------------

    // PCA fit
    val pca = new PCA(4).fit(vec.map(_.features))
    val pcaData = vec.map(p => p.copy(features = pca.transform(p.features)))

    // Load data to HDFS
    MLUtils.saveAsLibSVMFile(pcaData.coalesce(1), tempFile)
    //Unload data from HDFS
    data = spark.read.format("libsvm").load(tempFile)

    //--------------------- LR MODEL PERFORMANCE WITH PCA -----------------------------

    // Scaling
    val scaler_new = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)
    val scalerModel_new = scaler_new.fit(data)
    data = scalerModel_new.transform(data)

    // ML splits
    val Array(trainingData1, testData1) = data.randomSplit(Array(0.8, 0.2))

    // LinearRegression
    val lir_new = new LinearRegression().setFeaturesCol("scaledFeatures").setMaxIter(1000).setRegParam(0.01).setElasticNetParam(1.0)
    val model_new = lir_new.fit(trainingData1)
    val pred_new = model_new.evaluate(testData1)

    // Evaluation
    val evaluator_new = new RegressionEvaluator().setPredictionCol("prediction").setMetricName("rmse")
    val rmse2 = evaluator_new.evaluate(pred_new.predictions)

    //----------------------DISPLAY OUTPUT AND DELETE THE TEMP LIBSVM FILE-------------------------

    println("Linear Regression: Test Mean Squared Error = " + rmse1)
    println("Linear Regression with PCA: Test Mean Squared Error = " + rmse2)

    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val outPutPath = new Path(tempFile)

    if (fs.exists(outPutPath))
      fs.delete(outPutPath, true)

    System.exit(0)

  }}

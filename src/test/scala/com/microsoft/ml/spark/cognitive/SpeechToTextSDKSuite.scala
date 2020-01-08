// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cognitive

import java.io.{FileInputStream, FileNotFoundException}
import java.net.URI

import com.microsoft.ml.spark.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.commons.compress.utils.IOUtils
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.{DataFrame, Row}
import org.scalactic.Equality
import org.scalatest.Assertion

class SpeechToTextSDKSuite extends TransformerFuzzing[SpeechToTextSDK]
  with SpeechKey {

  import session.implicits._

  val region = "eastus"
  val resourcesDir = System.getProperty("user.dir") + "/src/test/resources/"
  val uri = new URI(s"https://$region.api.cognitive.microsoft.com/sts/v1.0/issuetoken")
  val language = "en-us"
  val profanity = "masked"
  val format = "simple"

  lazy val sdk = new SpeechToTextSDK()
    .setSubscriptionKey(speechKey)
    .setLocation(region)
    .setOutputCol("text")
    .setAudioDataCol("audio")
    .setLanguage("en-US")
    .setProfanity("Masked")

  lazy val audioPaths = Array[String](resourcesDir + "audio1.wav", resourcesDir + "audio2.wav")

  lazy val audioBytes: Array[Array[Byte]] = audioPaths.map(
    path => IOUtils.toByteArray(new FileInputStream((path)))
  )

  lazy val audioDfs: Array[DataFrame] = audioBytes.map(bytes =>
    Seq(Tuple1(bytes)).toDF("audio")
  )

  override lazy val dfEq = new Equality[DataFrame] {
    override def areEqual(a: DataFrame, b: Any): Boolean =
      baseDfEq.areEqual(a.drop("audio"), b.asInstanceOf[DataFrame].drop("audio"))
  }

  override def testSerialization(): Unit = {
    tryWithRetries(Array(0, 100, 100, 100, 100))(super.testSerialization)
  }

  /** Simple similarity test using Jaccard index */
  def jaccardSimilarity(s1: String, s2: String): Double = {
    val a = s1.toLowerCase.sliding(2).toSet
    val b = s2.toLowerCase.sliding(2).toSet
    a.intersect(b).size.toDouble / (a | b).size.toDouble
  }

  def speechTest(audioFile: String, textFile: String): Assertion = {
    val audioFilePath = resourcesDir + audioFile
    val expectedPath = resourcesDir + textFile
    val expectedFile = scala.io.Source.fromFile(expectedPath)
    val expected = try expectedFile.mkString finally expectedFile.close()
    val bytes = IOUtils.toByteArray(new FileInputStream(audioFilePath))
    val result = sdk.audioBytesToText(bytes, speechKey, uri, language, profanity, format)
    assert(jaccardSimilarity(expected, result) > .9)
  }

  def dfTest(format: String, audioFileNumber: Int, verbose: Boolean = false): Assertion = {
    val expectedFile = scala.io.Source.fromFile(resourcesDir + $"audio$audioFileNumber.txt")
    val expected = try expectedFile.mkString finally expectedFile.close()
    val result = sdk.setFormat(format)
      .transform(audioDfs(audioFileNumber-1)).select("text")
      .collect().head.getAs[String]("text")
    if (verbose) {
      println(s"Expected: $expected")
      println(s"Actual: $result")
    }
    assert(jaccardSimilarity(expected, result) > .9)
  }

  test("SDK audioBytesToText 1"){
    speechTest("audio1.wav", "audio1.txt")
  }

  test("SDK audioBytesToText 2"){
    speechTest("audio2.wav", "audio2.txt")
  }

  test("SDK audioBytesToText File Doesn't Exist") {
    assertThrows[FileNotFoundException] {
      speechTest("audio3.wav", "audio3.txt")
    }
  }

  test("Simple SDK Usage Audio 1") {
   dfTest("simple", 1, true)
  }

  test("Detailed SDK Usage Audio 1") {
    dfTest("detailed", 1, true)
  }

  test("Simple SDK Usage Audio 2") {
    dfTest("simple", 2, true)
  }

  test("Detailed SDK Usage Audio 2") {
    dfTest("detailed", 2, true)
  }

  test("API vs. SDK") {
    val stt = new SpeechToText()
      .setSubscriptionKey(speechKey)
      .setLocation(region)
      .setOutputCol("text")
      .setAudioDataCol("audio")
      .setLanguage("en-US")
    val toObj: Row => SpeechResponse = SpeechResponse.makeFromRowConverter
    val apiResult = toObj(stt.setFormat("simple")
      .transform(audioDfs(0)).select("text")
      .collect().head.getStruct(0)).DisplayText.getOrElse("")

    val sdkResult = sdk.setFormat(format)
      .transform(audioDfs(0)).select("text")
      .collect().head.getAs[String]("text")
    assert(jaccardSimilarity(apiResult, sdkResult) > 0.9)
  }

  override def testObjects(): Seq[TestObject[SpeechToTextSDK]] =
    Seq(new TestObject(sdk, audioDfs(1)))

  override def reader: MLReadable[_] = SpeechToTextSDK
}

// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cognitive

import com.microsoft.ml.spark.core.schema.SparkBindings

case class DetailedSpeechResponse(Confidence: Double,
                                  Lexical: String,
                                  ITN: String,
                                  MaskedITN: String,
                                  Display: String)

case class SpeechResponse(RecognitionStatus: String,
                          Offset: Int,
                          Duration: Int,
                          Id: Option[String],
                          DisplayText: Option[String],
                          NBest: Option[Seq[DetailedSpeechResponse]]
                          )

object SpeechResponse extends SparkBindings[SpeechResponse]

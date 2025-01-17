﻿/* from https://github.com/mdfarragher/DLR/tree/master/BinaryClassification/LstmDemo
 * Detect movie review sentiment */

using Microsoft.ML;
using Microsoft.ML.Data;
using CNTK;
using Sample.CNTKUtil;
using XPlot.Plotly;
using System;
using System.IO;
using System.Linq;
using System.IO.Compression;

namespace LstmDemo
{
    public class Program
    {
        private static readonly string _dataPath = @"C:\Users\dajmi\source\repos\ML\CNTK\LSTM\bin\Debug\net7.0\Data\imdb_data.zip";
        public static void Main(string[] args)
        {
            if (!File.Exists("x_train_imdb.bin"))
            {
                ZipFile.ExtractToDirectory(_dataPath, ".");
            }

            Console.WriteLine("Loading data files...");
            var sequenceLength = 500;
            var training_data = DataUtil.LoadBinary<float>("x_train_imdb.bin", 25000, sequenceLength);
            var training_labels = DataUtil.LoadBinary<float>("y_train_imdb.bin", 25000);
            var testing_data = DataUtil.LoadBinary<float>("x_test_imdb.bin", 25000, sequenceLength);
            var testing_labels = DataUtil.LoadBinary<float>("y_test_imdb.bin", 25000);

            var features = NetUtil.Var(new int[] { 1 }, CNTK.DataType.Float);
            var labels = NetUtil.Var(new int[] { 1 }, CNTK.DataType.Float,
                dynamicAxes: new List<CNTK.Axis>() { CNTK.Axis.DefaultBatchAxis() });

            // network building
            var lstmUnits = 5;
            var network = features
                //.OneHotOp(10000, true)
                //.Embedding(32)
                .LSTM(lstmUnits, lstmUnits)
                .Dense(1, CNTKLib.Sigmoid)
                .ToNetwork();
            Console.WriteLine("Model architecture:");
            Console.WriteLine(network.ToSummary());

            var lossFunc = CNTKLib.BinaryCrossEntropy(network.Output, labels);
            var errorFunc = NetUtil.BinaryClassificationError(network.Output, labels);

            // set up a learner
            var learner = network.GetSGDLearner(
                learningRateSchedule: (0.001, 1),
                momentumSchedule: (0.9, 1),
                unitGain: true);

            var trainer = network.GetTrainer(learner, lossFunc, errorFunc);
            var evaluator = network.GetEvaluator(errorFunc);

            // train the model
            Console.WriteLine("Epoch\tTrain\tTrain\tTest");
            Console.WriteLine("\tLoss\tError\tError");

            var maxEpochs = 10;
            var batchSize = 128;
            var loss = new double[maxEpochs];
            var trainingError = new double[maxEpochs];
            var testingError = new double[maxEpochs];
            var batchCount = 0;

            for (int epoch = 0; epoch < maxEpochs; epoch++)
            {
                loss[epoch] = 0.0;
                trainingError[epoch] = 0.0;
                batchCount = 0;
                training_data.Batch(batchSize, (data, begin, end) =>
                {
                    // get the current batch
                    var featureBatch = features.GetSequenceBatch(sequenceLength, training_data, begin, end);
                    var labelBatch = labels.GetBatch(training_labels, begin, end);

                    // train the network on the batch
                    var result = trainer.TrainBatch(
                        new[] {
                                (features, featureBatch),
                                (labels,  labelBatch)
                               },
                        false
                    );
                    loss[epoch] += result.Loss;
                    trainingError[epoch] += result.Evaluation;
                    batchCount++;
                });

                // show results
                loss[epoch] /= batchCount;
                trainingError[epoch] /= batchCount;
                Console.Write($"{epoch}\t{loss[epoch]:F3}\t{trainingError[epoch]:F3}\t");

                //test
                testingError[epoch] = 0.0;
                batchCount = 0;
                testing_data.Batch(batchSize, (data, begin, end) =>
                {
                    // get the current batch for testing
                    var featureBatch = features.GetSequenceBatch(sequenceLength, testing_data, begin, end);
                    var labelBatch = labels.GetBatch(testing_labels, begin, end);

                    // test the network on the batch
                    testingError[epoch] += evaluator.TestBatch(
                        new[] {
                            (features, featureBatch),
                            (labels,  labelBatch)
                             }
                    );
                    batchCount++;
                });
                testingError[epoch] /= batchCount;
                Console.WriteLine($"{testingError[epoch]:F3}");
            }

            // show final results
            var finalError = testingError[maxEpochs - 1];
            Console.WriteLine();
            Console.WriteLine($"Final test error: {finalError:0.00}");
            Console.WriteLine($"Final test accuracy: {1 - finalError:0.00}");

            /* plot goes here */
           // var chart = Chart.Plot(
           //    new[]
           //    {
           //         new Graph.Scatter()
           //         {
           //             x = Enumerable.Range(0, maxEpochs).ToArray(),
           //             y = trainingError.Select(v => 1 - v),
           //             name = "training",
           //             mode = "lines+markers"
           //         },
           //         new Graph.Scatter()
           //         {
           //             x = Enumerable.Range(0, maxEpochs).ToArray(),
           //             y = testingError.Select(v => 1 - v),
           //             name = "testing",
           //             mode = "lines+markers"
           //         }
           //    }
           //);
           // chart.WithOptions(new Layout.Layout()
           // {
           //     yaxis = new Graph.Yaxis()
           //     {
           //         rangemode = "tozero"
           //     }
           // });
           // chart.WithXTitle("Epoch");
           // chart.WithYTitle("Accuracy");
           // chart.WithTitle("Movie Review Sentiment");

           // // save chart
           // File.WriteAllText("chart.html", chart.GetHtml());
        }
    }
}

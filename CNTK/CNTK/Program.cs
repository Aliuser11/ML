//CNTK
// from : https://github.com/mdfarragher/DLR/tree/master/Regression/TaxiFarePrediction and https://github.com/mdfarragher/DLR/tree/master/CNTKUtil 

using Microsoft.ML;
using Microsoft.ML.Data;
using Sample.CNTKUtil;
using DataType = CNTK.DataType;

namespace TaxiFarePrediction
{

    public class TaxiTrip
    {
        [LoadColumn(0)] public float VendorId;
        [LoadColumn(5)] public float RateCode;
        [LoadColumn(3)] public float PassengerCount;
        [LoadColumn(4)] public float TripDistance;
        [LoadColumn(9)] public float PaymentType;
        [LoadColumn(10)] public float FareAmount;

        public float[] GetFeatures() => new float[] { VendorId, RateCode, PassengerCount, TripDistance, PaymentType };

        public float GetLabel() => FareAmount; // what we are predicting -> single taxi trip fare amount

        class Program
        {
            static readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "yellow_tripdata_2018-12_small.csv");

            static void Main()
            {
                var context = new MLContext();
                var textLoader = context.Data.CreateTextLoader(new TextLoader.Options()
                {
                    Separators = new[] { ',' },
                    HasHeader = true,
                    Columns = new[]
                    {
                        new TextLoader.Column("VendorId", DataKind.Single, 0),
                        new TextLoader.Column("RateCode", DataKind.Single, 5),
                        new TextLoader.Column("PassengerCount", DataKind.Single, 3),
                        new TextLoader.Column("TripDistance", DataKind.Single, 4),
                        new TextLoader.Column("PaymentType", DataKind.Single, 9),
                        new TextLoader.Column("FareAmount", DataKind.Single, 10)
                }});

                Console.Write("Loading training data...");
                var dataView = textLoader.Load(dataPath);

                var training = context.Data.CreateEnumerable<TaxiTrip>(dataView, reuseRowObject: false); //convert the data to an enumeration of TaxiTrip instances.

                // to use CNTK library float[][] is in need
                var trainData = training.Select(x => x.GetFeatures()).ToArray();
                var trainLabel = training.Select(x => x.GetLabel()).ToArray();

                //CNTK goes here
                var features = NetUtil.Var(new int[] { 5 }, DataType.Float);
                var label = NetUtil.Var(new int[] { 1 }, DataType.Float);

                //DENSE method to set up linear regression model
                var network = features
                                    .Dense(1)
                                    .ToNetwork();
                Console.WriteLine("Model architecture:");
                Console.WriteLine(network.ToSummary()); //output a description of the model

                //________________________________________________________________________________
                //use MSE as the loss function to measure error
                var lossFunc = NetUtil.MeanSquaredError(network.Output, label);
                //track the error with the MAE metric( measure in dollars !!)
                var errorFunc = NetUtil.MeanAbsoluteError(network.Output, label);

                //train and EVALUATE
                var learner = network.GetAdamLearner(
                                        learningRateSchedule: (0.001, 1),
                                        momentumSchedule: (0.9, 1),
                                        unitGain: false);

                var trainer = network.GetTrainer(learner, lossFunc, errorFunc);

                // train the model
                Console.WriteLine("Epoch\tTrain\tTrain");
                Console.WriteLine("\tLoss\tError");

                var maxEpochs = 50;
                var batchSize = 32;
                var loss = new double[maxEpochs];
                var trainingError = new double[maxEpochs];
                var batchCount = 0;
                for (int epoch = 0; epoch < maxEpochs; epoch++)
                {
                    // training and testing 
                    loss[epoch] = 0.0;
                    trainingError[epoch] = 0.0;
                    batchCount = 0;
                    trainData.Index().Shuffle().Batch(batchSize, (indices, begin, end) =>
                    {
                        //batch
                        var featureBatch = features.GetBatch(trainData, indices, begin, end);
                        var labelBatch = label.GetBatch(trainLabel, indices, begin, end);

                        // train network
                        var result = trainer.TrainBatch(
                            new[] {
                                    (features, featureBatch),
                                    (label,  labelBatch)
                                   },
                            false
                        );
                        loss[epoch] += result.Loss;
                        trainingError[epoch] += result.Evaluation;
                        batchCount++;
                    });

                    // results
                    loss[epoch] /= batchCount;
                    trainingError[epoch] /= batchCount;
                    Console.WriteLine($"{epoch}\t{loss[epoch]:F3}\t{trainingError[epoch]:F3}");
                }

                // final results
                var finalError = trainingError[maxEpochs - 1];
                Console.WriteLine();
                Console.WriteLine($"Final MAE: {finalError:0.00}");

            }
        }
    }
}
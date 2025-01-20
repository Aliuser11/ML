/* from: https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/predict-prices  */

using Microsoft.ML;
using ML.NET_Regression;
using Microsoft.ML.Data;

internal class Program
{
    private static void Main(string[] args)
    {
        string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        MLContext mlContext = new MLContext(seed: 0);
        var textLoader = mlContext.Data.CreateTextLoader(new TextLoader.Options()
        {
            Separators = new[] { ',' },
            HasHeader = true,
            Columns = new[]
            {   new TextLoader.Column("VendorId", DataKind.String, 0),
                new TextLoader.Column("RateCode", DataKind.String, 5),
                new TextLoader.Column("PassengerCount", DataKind.Single, 3),
                new TextLoader.Column("TripDistance", DataKind.Single, 4),
                new TextLoader.Column("TripTime", DataKind.Single, 4),
                new TextLoader.Column("PaymentType", DataKind.String, 9),
                new TextLoader.Column("FareAmount", DataKind.Single, 10) }
                });

        var model = Train(mlContext, _trainDataPath);
        Evaluate(mlContext, model);
        TestSinglePrediction(mlContext, model);


        ITransformer Train(MLContext mlContext, string dataPath)
        {

            //IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');
            IDataView dataView  = textLoader.Load(_trainDataPath);
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                                .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
                                .AppendCacheCheckpoint(mlContext)// cache data -> training speeds up
                                .Append(mlContext.Regression.Trainers.FastTree());

            var model = pipeline.Fit(dataView);
            return model;
        }

        void Evaluate(MLContext mlContext, ITransformer model)
        {
            //IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');
            IDataView dataView = textLoader.Load(_testDataPath);
            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:0.##}");
        }

        void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);
            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 3,
                TripTime = 20,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                //FareAmount = 0 // To predict. Actual/Observed = 15.5
            };

            var prediction = predictionFunction.Predict(taxiTripSample);
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine(prediction.FareAmount);
        }
    }
}
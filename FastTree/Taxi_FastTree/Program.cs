using Microsoft.ML;
using Microsoft.ML.Data;
using Taxi.Models;

string pathTrainingData = @"C:\Users\dajmi\source\repos\ML\FastTree\Taxi_FastTree\Data\taxi-fare-train.csv";
string pathTestData = @"C:\Users\dajmi\source\repos\ML\FastTree\Taxi_FastTree\Data\taxi-fare-test.csv";

char RetrainSeparatorChar = ',';
bool RetrainHasHeader = true;
bool RetrainAllowQuoting = false;
MLContext mlContext = new();
var dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(pathTrainingData, RetrainSeparatorChar, RetrainHasHeader, RetrainAllowQuoting);


var testDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(pathTestData, RetrainSeparatorChar, RetrainHasHeader, RetrainAllowQuoting);

//or use SPLIT ON DATA
//var partitions = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2); // 80/20


var pipeline = mlContext.Transforms.CopyColumns(
    inputColumnName: "FareAmount",
    outputColumnName: "Label")
    .Append(mlContext.Transforms.Categorical.OneHotEncoding("VendorId"))
    .Append(mlContext.Transforms.Categorical.OneHotEncoding("RateCode"))
    .Append(mlContext.Transforms.Categorical.OneHotEncoding("PaymentType"))
     .Append(mlContext.Transforms.Concatenate(
        "Features",
        "VendorId",
        "RateCode",
        "PassengerCount",
        "TripTime",
        "TripDistance",
        "PaymentType")) // one column
     .AppendCacheCheckpoint(mlContext)// cache data -> training speeds up
      .Append(mlContext.Regression.Trainers.FastTree()); //Model choice FAST TREE

var trainModel = pipeline.Fit(dataView);
var prediction = trainModel.Transform(testDataView);

var metrics = mlContext.Regression.Evaluate(prediction, "Label", "Score");
Console.WriteLine($"Model metrics:");
Console.WriteLine($"  RMSE:{metrics.RootMeanSquaredError:#.##}");
Console.WriteLine($"  MSE: {metrics.MeanSquaredError:#.##}");
Console.WriteLine($"  MAE: {metrics.MeanAbsoluteError:#.##}");
Console.WriteLine();

Console.WriteLine();
// Create prediction engine related to the loaded trained model
var predictionEngine = mlContext
.Model
.CreatePredictionEngine<TaxiTrip,
TaxiFarePrediction>(trainModel);


int i;
int pax = 0;
int dist = 0;
int time = 0;

//inputs
Console.WriteLine("How many passenger will you take?");
if (int.TryParse(Console.ReadLine(), out i))
{
    pax = i;
}

Console.WriteLine("Write trip distance in km.");
if (int.TryParse(Console.ReadLine(), out i))
{
    dist = i;
}

Console.WriteLine("Write trip time in min.");
if (int.TryParse(Console.ReadLine(), out i))
{
    time = i * 60;
}

var testTrip = new TaxiTrip()
{
    VendorId = "2",
    RateCode = "1",
    PassengerCount = pax,
    PaymentType = "1",
    TripDistance = dist,
    TripTime = time,
};

//predict
var predictedFee = predictionEngine.Predict(testTrip).FareAmount;
Console.WriteLine($"{predictedFee}");
Console.WriteLine($"Finished");

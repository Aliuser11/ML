using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

/*FROM: https://github.com/mdfarragher/DSC/tree/master/BinaryClassification/TitanicPrediction*/

// TRAIN and TEST Data
var PathForTrainData = Path.Combine(Environment.CurrentDirectory, @"C:\Users\dajmi\source\repos\ML\DecisionTrees\DecisionTrees_TitanicPrediction\Data\test_data.csv");
var PathForTestData = Path.Combine(Environment.CurrentDirectory, @"C:\Users\dajmi\source\repos\ML\DecisionTrees\DecisionTrees_TitanicPrediction\Data\train_data.csv");

MLContext context = new(); // set up machine learning context
var textLoader = context.Data.CreateTextLoader(new TextLoader.Options()
{
    Separators = new[] { ',' },
    HasHeader = true,
    AllowQuoting = true,
    Columns = new[]
        {
            new TextLoader.Column("Label",DataKind.Boolean,1),
            new TextLoader.Column("Pclass", DataKind.Single, 2),
            new TextLoader.Column("Name", DataKind.String, 3),
            new TextLoader.Column("Sex", DataKind.String, 4),
            new TextLoader.Column("RawAge", DataKind.String, 5),
            new TextLoader.Column("SibSp", DataKind.Single, 6),
            new TextLoader.Column("Parch", DataKind.Single, 7),
            new TextLoader.Column("Ticket", DataKind.String, 8),
            new TextLoader.Column("Fare", DataKind.Single, 9),
            new TextLoader.Column("Cabin", DataKind.String, 10),
            new TextLoader.Column("Embarked", DataKind.String, 11)
        }
});

Console.WriteLine($"Loading data...");
var dataView = textLoader.Load(PathForTrainData); // load csv data
var dataViewForTesting = textLoader.Load(PathForTestData); //load csv data

/*formating the data*/
// replace empty age column  with <?>
var pipeline = context.Transforms.DropColumns("Name", "Cabin", "Ticket")
                                 .Append(context.Transforms.CustomMapping<FromAge, ToAge>(
                                  (i, o) => { o.Age = string.IsNullOrEmpty(i.RawAge) ? "?" : i.RawAge; }, "AgeMapping"))
                                  .Append(context.Transforms.Conversion.ConvertType("Age", outputKind: DataKind.Single)) //change type of Age column to float -> Single =  single-precision floating point valu
                                  .Append(context.Transforms.ReplaceMissingValues("Age", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)) //relace missing age with mean value
                                  .Append(context.Transforms.Categorical.OneHotEncoding("Sex")) //one-hot encode , return new column that holds one-hot vector
                                  .Append(context.Transforms.Categorical.OneHotEncoding("Embarked"))
                                  .Append(context.Transforms.Concatenate("Features",
                                                                            "Age",
                                                                            "Pclass",
                                                                            "SibSp",
                                                                            "Parch",
                                                                            "Sex",
                                                                            "Embarked"))
                                  //use choosen trainning model
                                  .Append(context.BinaryClassification.Trainers.FastTree(
                                      labelColumnName: "Label",
                                      featureColumnName: "Features"));

Console.WriteLine(  $"Training the model");
var trainedModel = pipeline.Fit(dataView);
// make prediction for test data
Console.WriteLine(  $"Evaluate.. make prediction for test data");
var prediction = trainedModel.Transform(dataViewForTesting);
//compare output
var metrics = context.BinaryClassification.Evaluate(
    data: prediction,
    labelColumnName: "Label",
    scoreColumnName: "Score");
//results:
Console.WriteLine($"  Accuracy: correct/allPredictions -> {metrics.Accuracy:P2}"); 
Console.WriteLine($"  Auc: 0 wrong alltime| 0,5 random output| 1 correct alltime -> {metrics.AreaUnderRocCurve:P2}");
Console.WriteLine($"  Auprc:{metrics.AreaUnderPrecisionRecallCurve:P2}");
Console.WriteLine($"  F1Score: {metrics.F1Score:P2}");
Console.WriteLine($"  LogLoss: {metrics.LogLoss:0.##}");
Console.WriteLine($"  LogLossReduction: {metrics.LogLossReduction:0.##}");
Console.WriteLine($"  PositivePrecision fraction of positive predictions that are correct: {metrics.PositivePrecision:0.##}");
Console.WriteLine($"  PositiveRecall fraction of positive predictions of all cases: {metrics.PositiveRecall:0.##}");
Console.WriteLine($"  NegativePrecision: {metrics.NegativePrecision:0.##}");
Console.WriteLine($"  NegativeRecall: {metrics.NegativeRecall:0.##}");
Console.WriteLine();

// predicttion for single value
var predictPassengerEngine = context.Model.CreatePredictionEngine<Passenger, PassengerPrediction>(trainedModel);
var pass1 = new Passenger()
{
    Pclass = 1,
    Name = "XX XXX",
    Sex = "male",
    RawAge = "68",
    SibSp = 0,
    Parch = 0,
    Fare = 70,
    Embarked = "S"
};

var singlePrediction = predictPassengerEngine.Predict(pass1);
Console.WriteLine($"Passenger:   {pass1.Name},\n  Prediction:  {(singlePrediction.Prediction ? "survived" : "perished")}, \n Probability: {singlePrediction.Probability}");
Console.WriteLine(  $"Finished : )");
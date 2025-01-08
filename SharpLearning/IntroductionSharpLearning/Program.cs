/* from https://github.com/mdabros/SharpLearning/wiki/Introduction-to-SharpLearning */

using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using SharpLearning.RandomForest.Learners;
using SharpLearning.RandomForest.Models;
using SharpLearning.Common.Interfaces;
using SharpLearning.InputOutput.Serialization;

var parser = new CsvParser(() => new StreamReader("whiteWineQuality.csv"));
var target = "quality";

var observations = parser.EnumerateRows(x => x != target).ToF64Matrix(); //converts from CsvRows to double format.
var targets = parser.EnumerateRows(target).ToF64Vector(); //converts from CsvRows to double format.

// split to test and train data using RandomTrainingTestIndexSplitter |  30 % of the data is used for the test set. 
var splitter = new RandomTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24);

var trainingTestSplit = splitter.SplitSet(observations, targets);
var trainSet = trainingTestSplit.TrainingSet;
var testSet = trainingTestSplit.TestSet;

/*  there are  metrics available in SharpLearning.Metrics. A standard metric for evaluating a regression -> mean square error. */
var learner = new RegressionRandomForestLearner(trees: 100);
var model = learner.Learn(trainSet.Observations, trainSet.Targets);

// predict training and test set.
var trainPredictions = model.Predict(trainSet.Observations);
var testPredictions = model.Predict(testSet.Observations);

var metric = new MeanSquaredErrorRegressionMetric();
var trainError = metric.Error(trainSet.Targets, trainPredictions);
var testError = metric.Error(testSet.Targets, testPredictions);

// the variable importance requires the featureNameToIndex from the data set. This mapping describes the relation from column name to index in the feature matrix.
var featureNameToIndex = parser.EnumerateRows(c => c != target).First().ColumnNameToIndex;

// Get the variable importance from the model.
var importances = model.GetVariableImportance(featureNameToIndex);

Console.WriteLine($"importances: ", '\t');
foreach (var importance in importances)
{
    Console.WriteLine( $" {importance.Key} -> { importance.Value}");
}
                    /* output will be:
                     "[alcohol, 100]"
                    "[density, 53.52123936525723]"
                    "[chlorides, 31.374691164220714]"
                    "[volatile acidity, 25.551870161741626]"
                    "[free sulfur dioxide, 18.13027432317995]"
                    "[total sulfur dioxide, 13.760638388186713]"
                    "[citric acid, 11.159767098928468]"
                    "[residual sugar, 5.928354475582021]"
                    "[pH, 4.975183189344861]"
                    "[fixed acidity, 3.5771215648809322]"
                    "[sulphates, 2.49386847677607]"
                     */

/*  SAVE AND LOAD 
 *  -> o p t i o n s */

//save and load model
model.Save(() => new StreamWriter(@"randomforest.xml"));
var loadedModel = RegressionForestModel.Load(() => new StreamReader(@"randomforest.xml"));

/*
//using serializer
var xmlSerializer = new GenericXmlDataContractSerializer();
xmlSerializer.Serialize<IPredictorModel<double>>(model, () => new StreamWriter(@"randomforest.xml"));
var loadedModelXml = xmlSerializer.Deserialize<IPredictorModel<double>>(() => new StreamReader(@"randomforest.xml"));

//using GenericBinarySerializer
var binarySerializer = new GenericBinarySerializer();
binarySerializer.Serialize<IPredictorModel<double>>(model, () => new StreamWriter(@"C:\randomforest.bin"));
var loadedModelBinary = binarySerializer.Deserialize<IPredictorModel<double>>(() => new StreamReader(@"C:\randomforest.bin"));*/
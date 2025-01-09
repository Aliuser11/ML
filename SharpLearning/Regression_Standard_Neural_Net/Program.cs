/* from https://github.com/mdabros/SharpLearning.Examples/blob/master/src/SharpLearning.Examples/NeuralNets/RegressionNeuralNetExamples.cs */

using System.Diagnostics;
using SharpLearning.Containers.Matrices;
using SharpLearning.FeatureTransformations.MatrixTransforms;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using SharpLearning.Neural;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Layers;
using SharpLearning.Neural.Learners;
using SharpLearning.Neural.Loss;

namespace SharpLearning
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //using StreamReader to read filepath
            var parser = new CsvParser(() => new StreamReader("whiteWineQuality.csv"));
            var target = "quality";

            var targets = parser.EnumerateRows(target).ToF64Vector();
            var observations = parser.EnumerateRows(x => x != target).ToF64Matrix();

            //transform values to be between 0 - 1
            var minMaxTransformer = new MinMaxTransformer(0.0, 1.0);
            minMaxTransformer.Transform(observations, observations);
            var numberOfFeatures = observations.ColumnCount;

            var net = new NeuralNet();
            net.Add(new InputLayer(inputUnits: numberOfFeatures));
            net.Add(new DropoutLayer(0.2));
            net.Add(new DenseLayer(800, Activation.Relu));
            net.Add(new DropoutLayer(0.5));
            net.Add(new DenseLayer(800, Activation.Relu));
            net.Add(new DropoutLayer(0.5));
            net.Add(new SquaredErrorRegressionLayer());  //square error as error metric.

            var learner = new RegressionNeuralNetLearner(net, iterations: 10, loss: new SquareLoss());
            var model = learner.Learn(observations, targets);
            var metric = new MeanSquaredErrorRegressionMetric();
            var prediction = model.Predict(observations);
            Trace.WriteLine("training error: " + metric.Error(targets, prediction));
            Console.WriteLine("training error: " + metric.Error(targets, prediction));
        }
    }
}
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math.Optimization.Losses;
using Linear = Accord.Statistics.Kernels.Linear;

namespace Regression
{
    class LinearSvm
    {
        internal void LinearSvm1()
        {
            // Declare a very simple regression problem with only 2 input variables (x and y):
            double[][] inputs =
            {
                new[] { 3.0, 1.0 },
                new[] { 7.0, 1.0 },
                new[] { 3.0, 1.0 },
                new[] { 3.0, 2.0 },
                new[] { 6.0, 1.0 },
            };

            // The task is to output a weighted sum of those numbers   plus an independent constant term: 7.4x + 1.1y + 42
            double[] outputs =
            {
                7.4*3.0 + 1.1*1.0 + 42.0,
                7.4*7.0 + 1.1*1.0 + 42.0,
                7.4*3.0 + 1.1*1.0 + 42.0,
                7.4*3.0 + 1.1*2.0 + 42.0,
                7.4*6.0 + 1.1*1.0 + 42.0,
            };

            // Create a new Sequential Minimal Optimization (SMO) learning  algorithm and estimate the complexity parameter C from data
            var teacher = new SequentialMinimalOptimization<Linear>() //SMO
            {
                UseComplexityHeuristic = true,
                Complexity = 100000.0 // Note: do not do this in an actual application!
            };

            var svm = teacher.Learn(inputs, outputs);

            // Classify the samples using the model
            double[] answers = svm.Score(inputs); // 1,1,1,1,1
            double error = new SquareLoss(outputs).Loss(answers); // should be 0.0
        }

        internal void LinearSvm2()
        {
            double[][] inputs =
            {
                new[] { 3.0, 1.0 },
                new[] { 7.0, 1.0 },
                new[] { 3.0, 1.0 },
                new[] { 3.0, 2.0 },
                new[] { 6.0, 1.0 },
            };

            // 7.4x + 1.1y + 42
            double[] outputs =
            {
                7.4*3.0 + 1.1*1.0 + 42.0,
                7.4*7.0 + 1.1*1.0 + 42.0,
                7.4*3.0 + 1.1*1.0 + 42.0,
                7.4*3.0 + 1.1*2.0 + 42.0,
                7.4*6.0 + 1.1*1.0 + 42.0,
            };

            // Create Newton-based support vector regression 
            var teacher = new LinearRegressionNewtonMethod() // NEWTON
            {
                Tolerance = 1e-5,
                Complexity = 10000
            };
            var svm = teacher.Learn(inputs, outputs);

            double[] prediction = svm.Score(inputs);

            double error = new SquareLoss(outputs).Loss(prediction); // Compute the error in the prediction (should be 0.0)
            Console.WriteLine(error);
        }
    }
}
    
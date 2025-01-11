using Accord.Math.Optimization.Losses;
using Accord.Statistics.Models.Regression.Linear;

namespace LinearRegression
{
    class MLR
    {
        internal void MultipleLinearRegressions()
        //ax + by + c = z 
        {
            double[][] inputs =
            {
                new double[] { 1, 1 },
                new double[] { 0, 1 },
                new double[] { 1, 0 },
                new double[] { 0, 0 },
            };

            // located in the same Z (z = 1)
            double[] outputs = { 1, 1, 1, 1 };
            var ols = new OrdinaryLeastSquares()
            {
                UseIntercept = true
            };

            MultipleLinearRegression regression = ols.Learn(inputs, outputs); // regression model

            //outputs  ->  ax + by + c = z => 0x + 0y + 1 = z => 1 = z.
            double a = regression.Weights[0]; // a = 0
            double b = regression.Weights[1]; // b = 0
            double c = regression.Intercept;  // c = 1

            // We can compute the predicted points using
            double[] predicted = regression.Transform(inputs);

            // And the squared error loss using 
            double error = new SquareLoss(outputs).Loss(predicted);
        }

    }
}
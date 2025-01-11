using Accord.Math.Optimization.Losses;
using Accord.Statistics.Models.Regression.Linear;

namespace LinearRegression
{
    class LinearMulti
    {
        internal void MultivariateLinear()
        {
            double[][] inputs =
            {
                // variables:  x1  x2  x3
                new double[] {  1,  1,  1 }, // input sample 1
                new double[] {  2,  1,  1 }, // input sample 2
                new double[] {  3,  1,  1 }, // input sample 3
            };

            double[][] outputs =
            {
                // variables:  y1  y2
                new double[] {  2,  3 }, // corresponding output to sample 1
                new double[] {  4,  6 }, // corresponding output to sample 2
                new double[] {  6,  9 }, // corresponding output to sample 3
            };

            // Use Ordinary Least Squares to create the regression
            OrdinaryLeastSquares ols = new OrdinaryLeastSquares();

            // Now, compute the multivariate linear regression:
            MultivariateLinearRegression regression = ols.Learn(inputs, outputs);

            // We can obtain predictions using
            double[][] predictions = regression.Transform(inputs);

            // The prediction error is
            double error = new SquareLoss(outputs).Loss(predictions); // 0
        }
    }
}
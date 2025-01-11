using Accord.Statistics.Models.Regression.Linear;

namespace LinearRegression
{
    class Linear
    {
        internal void LinearRegression1()
        {
            double[] inp = { 80, 60, 10, 20, 30 };
            double[] outp = { 20, 40, 30, 50, 60 };

            OrdinaryLeastSquares ols = new OrdinaryLeastSquares();
            SimpleLinearRegression regression = ols.Learn(inp, outp);

            double answer = regression.Transform(85); // The answer will be 28.088

            //other
            double s = regression.Slope;     // -0.264706
            double c = regression.Intercept; // 50.588235
        }
    }
}

using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Kernels;
using Accord.Math.Optimization;

// from -> https://github.com/accord-net/framework/blob/development/Samples/Tutorials/2.%20Regression/2.1.%20Non-linear/Program.cs

double[][] inputs =
{
    new[] { 3.0, 1.0 },
    new[] { 7.0, 1.0 },
    new[] { 3.0, 1.0 },
    new[] { 3.0, 2.0 },
    new[] { 6.0, 1.0 },
};

// search for output to a non linear combination ->  log(7.4x) / sqrt(1.1y + 42)
double[] outputs =
{
Math.Log(7.4*3.0) / Math.Sqrt(1.1*1.0 + 42.0),
Math.Log(7.4*7.0) / Math.Sqrt(1.1*1.0 + 42.0),
Math.Log(7.4*3.0) / Math.Sqrt(1.1*1.0 + 42.0),
Math.Log(7.4*3.0) / Math.Sqrt(1.1*2.0 + 42.0),
Math.Log(7.4*6.0) / Math.Sqrt(1.1*1.0 + 42.0),
};

// 3 solving options
var predictionSvm1 = KernelSvm1(inputs, outputs);
var predictionSvm2 = KernelSvm2(inputs, outputs);
var opt = Optimization(inputs, outputs);


double[] KernelSvm1(double[][] inputs, double[] outputs)
{
    var teacher = new FanChenLinSupportVectorRegression<Gaussian>()
    {
        Tolerance = 1e-5,
        Complexity = 10000,
        Kernel = new Gaussian(0.1)
    };

    var svm = teacher.Learn(inputs, outputs);//learn machine
    double[] prediction = svm.Score(inputs);//predict
    double error = new SquareLoss(outputs).Loss(prediction);//calculate error

    Console.WriteLine(error);
    return prediction;
    /*
     0.4732108783996186
    0.6002726061278392
    0.4732108783996186
    0.4672979306957006
    0.5767921348566217
     */
}

double[] KernelSvm2(double[][] inputs, double[] outputs)
{
    var teacher = new SequentialMinimalOptimization<Gaussian>()
    {
        UseComplexityHeuristic = true,
        UseKernelEstimation = true // estimate the kernel from the data
    };
    
    var svm = teacher.Learn(inputs, outputs);//learn machine
    double[] answers = svm.Score(inputs);//predict
    double error = new SquareLoss(outputs).Loss(answers);//calculate error
    Console.WriteLine(error);
    return answers;
    /*
        1
        1
        1
        1
        1
    */
}

double[] Optimization(double[][] inputs, double[] outputs)
{
    // Non-linear regression can also be solved using arbitrary models
    // log(w0  * x) / sqrt(w1 * y + w2)
    Func<double[], double[], double> model = (double[] x, double[] w) => Math.Log(w[0] * x[0]) / Math.Sqrt(w[1] * x[1] + w[2]); 

    //find the best parameters w that minimizes the error :

    Func<double[], double> objective = (double[] w) =>
    {
        double sumOfSquares = 0.0;
        for (int i = 0; i < inputs.Length; i++)
        {
            double expected = outputs[i];
            double actual = model(inputs[i], w);
            sumOfSquares += Math.Pow(expected - actual, 2);
        }
        return sumOfSquares;
    };

    // gradient free optimization algorithm 
    var cobyla = new Cobyla(numberOfVariables: 3) // 3 parameters:
    {
        Function = objective,
        MaxIterations = 100,
        Solution = new double[] { 15, 36.4, 101 } 
    };

    bool success = cobyla.Minimize(); // ??
    double[] solution = cobyla.Solution;

    double[] prediction = inputs.Apply(x => model(x, solution));//predict
    double error = new SquareLoss(outputs).Loss(prediction);//0
    Console.WriteLine(error);
    return prediction;
}

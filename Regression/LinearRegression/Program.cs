using LinearRegression;
using Linear = LinearRegression.Linear;
using Regression;

/* from: https://github.com/accord-net/framework/blob/development/Samples/Tutorials/2.%20Regression/2.1.%20Linear/Program.cs */

var linearRegression = new Linear();
linearRegression.LinearRegression1();

var mlr = new MLR();
mlr.MultipleLinearRegressions();

var linearSvm = new LinearSvm();
linearSvm.LinearSvm1();
linearSvm.LinearSvm2();

var hazard = new Hazard();
hazard.CoxProportionalHazards();

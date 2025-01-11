using Accord.Math;
using Accord.Statistics.Distributions.Univariate;
using Accord.Statistics.Models.Regression.Fitting;
using Accord.Statistics.Models.Regression;

namespace LinearRegression
{
    class Hazard
    {
        internal void CoxProportionalHazards()
        {

            object[,] data =
            {
                //    input         time until           outcome 
                // (features)     event happened     (what happened?)
                {       50,              1,         SurvivalOutcome.Censored  },
                {       70,              2,         SurvivalOutcome.Failed    },
                {       45,              3,         SurvivalOutcome.Censored  },
                {       35,              5,         SurvivalOutcome.Censored  },
                {       62,              7,         SurvivalOutcome.Failed    },
                {       50,             11,         SurvivalOutcome.Censored  },
                {       45,              4,         SurvivalOutcome.Censored  },
                {       57,              6,         SurvivalOutcome.Censored  },
                {       32,              8,         SurvivalOutcome.Censored  },
                {       57,              9,         SurvivalOutcome.Failed    },
                {       60,             10,         SurvivalOutcome.Failed    },
            }; 

            double[][] inputs = data.GetColumn(0).ToDouble().ToJagged();
            double[] time = data.GetColumn(1).ToDouble();
            SurvivalOutcome[] output = data.GetColumn(2).To<SurvivalOutcome[]>();

            var teacher = new ProportionalHazardsNewtonRaphson()//PH Newton-Raphson learning algorithm
            {
                ComputeBaselineFunction = true,
                ComputeStandardErrors = true,
                MaxIterations = 100
            };
            ProportionalHazards regression = teacher.Learn(inputs, time, output);// Proportional Hazards regression model

            SurvivalOutcome[] prediction = regression.Decide(inputs);//prediction

            double[] score = regression.Score(inputs);//score estimates

            double[] probability = regression.Probability(inputs);//probability estimates
        }
    }
}
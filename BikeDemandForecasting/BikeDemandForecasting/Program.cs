using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using System.Data.SqlClient;
using System.Reflection;

namespace BikeDemandForecasting
{
    // from -> https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/time-series-demand-forecasting
    internal class Program
    {
        private static void Main(string[] args)
        {
            string rootDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "../../../"));
            string dbFilePath = Path.Combine(rootDir, "Data", "DailyDemand.mdf");
            string modelPath = Path.Combine(rootDir, "MLModel.zip");
            var connectionString = $"Data Source=(LocalDB)\\MSSQLLocalDB;AttachDbFilename={dbFilePath};Integrated Security=True;Connect Timeout=30;";


            //initialize mlContext
            MLContext mlContext = new MLContext();

            //load data from db
            DatabaseLoader loader = mlContext.Data.CreateDatabaseLoader<ModelInput>();
            string query = "SELECT RentalDate, CAST(Year as REAL) as Year, CAST(TotalRentals as REAL) as TotalRentals FROM Rentals";

            //DatabaseSource
            DatabaseSource dbSource = new DatabaseSource(SqlClientFactory.Instance,
                                            connectionString,
                                            query);

            //IDataView
            IDataView dataView = loader.Load(dbSource);

            //filter data for training and testing
            IDataView firstYearData = mlContext.Data.FilterRowsByColumn(dataView, "Year", upperBound: 1);
            IDataView secondYearData = mlContext.Data.FilterRowsByColumn(dataView, "Year", lowerBound: 1);


            //pipeline using  SsaForecastingEstimator
            var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
                outputColumnName: "ForecastedRentals",
                inputColumnName: "TotalRentals",
                windowSize: 7,
                seriesLength: 30,
                trainSize: 365,
                horizon: 7,
                confidenceLevel: 0.95f,
                confidenceLowerBoundColumn: "LowerBoundRentals",
                confidenceUpperBoundColumn: "UpperBoundRentals");

            //fit method -> training
            SsaForecastingTransformer forecaster = forecastingPipeline.Fit(firstYearData);


            // evaluate model
            Evaluate(secondYearData, forecaster, mlContext);

            // save  -> TimeSeriesPredictionEngine. single prediction
            var forecastEngine = forecaster.CreateTimeSeriesEngine<ModelInput, ModelOutput>(mlContext);

            //checkpoint method to save
            forecastEngine.CheckPoint(mlContext, modelPath);

            //forecast
            Forecast(secondYearData, 7, forecastEngine, mlContext);

        }

       static void  Evaluate(IDataView testData, ITransformer model, MLContext mlContext)
        {
            // using transform  method
            IDataView predictions = model.Transform(testData);

            //actual values
            IEnumerable<float> actual =
            mlContext.Data.CreateEnumerable<ModelInput>(testData, true)
                .Select(observed => observed.TotalRentals);

            //forecast values 
            IEnumerable<float> forecast =
            mlContext.Data.CreateEnumerable<ModelOutput>(predictions, true)
                .Select(prediction => prediction.ForecastedRentals[0]);


            //error -> the difference
            var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);

            //measure performance
            var MAE = metrics.Average(error => Math.Abs(error)); // Mean Absolute Error
            var RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2))); // Root Mean Squared Error

        }
        static  void Forecast(IDataView testData, int horizon, TimeSeriesPredictionEngine<ModelInput, ModelOutput> forecaster, MLContext mlContext)
        {
            //predict forecast rentals
            ModelOutput forecast = forecaster.Predict();

            IEnumerable<string> forecastOutput =
                mlContext.Data.CreateEnumerable<ModelInput>(testData, reuseRowObject: false)
                    .Take(horizon)
                    .Select((ModelInput rental, int index) =>
                    {
                        string rentalDate = rental.RentalDate.ToShortDateString();
                        float actualRentals = rental.TotalRentals;
                        float lowerEstimate = Math.Max(0, forecast.LowerBoundRentals[index]);
                        float estimate = forecast.ForecastedRentals[index];
                        float upperEstimate = forecast.UpperBoundRentals[index];
                        return $"Date: {rentalDate}\n" +
                        $"Actual Rentals: {actualRentals}\n" +
                        $"Lower Estimate: {lowerEstimate}\n" +
                        $"Forecast: {estimate}\n" +
                        $"Upper Estimate: {upperEstimate}\n";
                    });

            // iteration and console visualization
            Console.WriteLine("Rental Forecast");
            Console.WriteLine("---------------------");
            foreach (var prediction in forecastOutput)
            {
                Console.WriteLine(prediction);
            }
        }
    }
}
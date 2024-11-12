using Microsoft.ML.Data;

public class PassengerPrediction
{
    [ColumnName("PredictedLabel")] public bool Prediction;
    public float Probability;
    public float Score;
}
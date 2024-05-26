using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

// from: https://blog.jetbrains.com/dotnet/2023/02/01/getting-started-with-ml-dotnet-machine-learning/
// data: https://raw.githubusercontent.com/khalidabuhakmeh/MachineLearningDotnet7/main/MachineLearning/yelp_labelled.txt

MLContext ctx = new MLContext();
ITransformer model;

if (File.Exists("model.zip"))
{
    // load from disk
    Console.WriteLine("Loading model from disk...");
    model = ctx.Model.Load("model.zip", out DataViewSchema schema);
}
else
{
    // load data
    Console.WriteLine("Loading yelp_labelled.txt...");
    IDataView dataView = ctx.Data
        .LoadFromTextFile<SentimentData>("yelp_labelled.txt");

    // split data into testing set
    DataOperationsCatalog.TrainTestData splitDataView = ctx.Data
        .TrainTestSplit(dataView, testFraction: 0.2);

    // Build model
    EstimatorChain<
        BinaryPredictionTransformer<
            CalibratedModelParametersBase<
                LinearBinaryModelParameters, PlattCalibrator>>> estimator = ctx.Transforms.Text
        .FeaturizeText(
            outputColumnName: "Features",
            inputColumnName: nameof(SentimentData.Text)
        ).Append(ctx.BinaryClassification.Trainers.SdcaLogisticRegression(featureColumnName: "Features"));

    // training happens here
    Console.WriteLine("Training model...");
    model = estimator.Fit(splitDataView.TrainSet);

    // evaluate the accuracy of our model
    IDataView predictions = model.Transform(splitDataView.TestSet);
    CalibratedBinaryClassificationMetrics metrics = ctx.BinaryClassification.Evaluate(predictions);
    Console.WriteLine($"Accuracy: {metrics.Accuracy:P00}");
    Console.WriteLine($"Area Under Roc Curve: {metrics.AreaUnderRocCurve:P00}");
    Console.WriteLine($"F1 Score: {metrics.F1Score:P00}");

    // save to disk
    Console.WriteLine("Saving model to disk...");
    ctx.Model.Save(model, dataView.Schema, "model.zip");
}

PredictionEngine<SentimentData, SentimentPrediction> engine =
    ctx.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

RunPrediction(engine, "this is a great restaurant");
RunPrediction(engine, "this is a bad restaurant");
return;

void RunPrediction(PredictionEngine<SentimentData, SentimentPrediction> engine, string text)
{
    SentimentData input = new SentimentData { Text = text };
    SentimentPrediction result = engine.Predict(input);

    Console.WriteLine($"{text} ({result.Probability:P00})");
}

class SentimentData
{
    [LoadColumn(0)] public string? Text;
    [LoadColumn(1), ColumnName("Label")] public bool Sentiment;
}

class SentimentPrediction : SentimentData
{
    [ColumnName("PredictedLabel")] public bool Prediction { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}
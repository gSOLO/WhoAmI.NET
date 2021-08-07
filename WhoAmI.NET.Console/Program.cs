using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

namespace WhoAmI.NET.Console
{
    class Program
    {
        static readonly String DATA_PATH = "..\\..\\..\\..\\Data\\data.csv";
        static void Main(string[] args)
        {
            System.Console.WriteLine(DATA_PATH);

            var context = new MLContext(seed: 10);

            var data = context.Data.LoadFromTextFile<Input>(DATA_PATH, separatorChar: ',', hasHeader: true);
            data = context.Data.Cache(data);

            var splitData = context.Data.TrainTestSplit(data, testFraction: 0.2, seed: 0);
            var trainData = splitData.TrainSet;
            var testData = splitData.TestSet;

            System.Console.WriteLine("Training The Model");

            var pipeline = context.Transforms.Conversion.MapValueToKey("Label")
                .Append(context.Transforms.Text.FeaturizeText("Features", "Text"))
                .Append(context.MulticlassClassification.Trainers.SdcaNonCalibrated())
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            var model = pipeline.Fit(trainData);

            var predicitons = model.Transform(testData);
            var metrics = context.MulticlassClassification.Evaluate(predicitons, "Label");

            System.Console.WriteLine($"Macro Accuracy: {(metrics.MacroAccuracy * 100):0.##}%");
            System.Console.WriteLine($"Micro Accuracy: {(metrics.MicroAccuracy * 100):0.##}%");
            System.Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            System.Console.WriteLine();

            var predictor = context.Model.CreatePredictionEngine<Input, Output>(model);

            while (true)
            {
                System.Console.WriteLine("Who Am I?");
                var line = System.Console.ReadLine();
                var prediction = predictor.Predict(new Input() { Text = line });

                System.Console.WriteLine($"Prediction: {prediction.Name} ({prediction.Score})");
                System.Console.WriteLine();
            }
        }
    }

    public class Input
    {
        [LoadColumn(0), ColumnName("Label")]
        public String Name { get; set; }

        [LoadColumn(1)]
        public String Text { get; set; }
    }

    public class Output
    {
        [ColumnName("PredictedLabel")]
        public String Name { get; set; }
        public float[] Score { get; set; }
    }
}

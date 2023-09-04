using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using SharpLearning.RandomForest.Learners;

namespace WhiteWineTesting;

public class Program
{
    public static void Main(string[] args)
    {
        var parser = new CsvParser(() => new StreamReader(Application.StartupPath + "\\winequality-white.csv"));
        var targetName = "quality";

        // read feature matrix
        var observations = parser.EnumerateRows(c => c != targetName).ToF64Matrix();

        // read regression targets
        var targets = parser.EnumerateRows(targetName).ToF64Vector();

        // Since this is a regression problem, we use the random training/test set splitter.
        // 30 % of the data is used for the test set. 
        var splitter = new RandomTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24);

        var trainingTestSplit = splitter.SplitSet(observations, targets);
        var trainSet = trainingTestSplit.TrainingSet;
        var testSet = trainingTestSplit.TestSet;


        var learner = new RegressionRandomForestLearner(trees: 100);
        var model = learner.Learn(trainSet.Observations, trainSet.Targets);
        var trainPredictions = model.Predict(trainSet.Observations);
        var testPredictions = model.Predict(testSet.Observations);

        // since this is a regression problem we are using square error as metric
        // for evaluating how well the model performs.
        var metric = new MeanSquaredErrorRegressionMetric();

        // measure the error on training and test set.
        var trainError = metric.Error(trainSet.Targets, trainPredictions);
        var testError = metric.Error(testSet.Targets, testPredictions);

        Trace.WriteLine($"Train error: {trainError:0.0000} - Test error: {testError:0.0000}");
        System.Console.WriteLine($"Train error: {trainError:0.0000} - Test error: {testError:0.0000}");


        // the variable importance requires the featureNameToIndex from the data set. This mapping describes the relation
        // from column name to index in the feature matrix.
        var featureNameToIndex = parser.EnumerateRows(c => c != targetName).First().ColumnNameToIndex;

        // Get the variable importance from the model.
        // Variable importance is a measure made by to model of how important each feature is.
        var importances = model.GetVariableImportance(featureNameToIndex);

        // trace normalized importances as csv.
        var importanceCsv = new StringBuilder();
        importanceCsv.Append("FeatureName;Importance");
        System.Console.WriteLine("FeatureName\tImportance");
        foreach (var feature in importances)
        {
            importanceCsv.AppendLine();
            importanceCsv.Append($"{feature.Key};{feature.Value:0.00}");
            System.Console.WriteLine($"{feature.Key}\t{feature.Value:0.00}");
        }

        Trace.WriteLine(importanceCsv);
        System.Console.ReadKey();
    }


    static void TraceTrainingAndTestError(double trainError, double testError)
    {
        Trace.WriteLine($"Train error: {trainError:0.0000} - Test error: {testError:0.0000}");
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using DTRFSample.Properties;
using SharpLearning.Containers.Extensions;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using SharpLearning.RandomForest.Learners;

namespace DTRFSample
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.ReadKey();
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var rows = targets.Length;
            int trees = 100;
            double subSample = 1.0;

            var sut = new RegressionExtremelyRandomizedTreesLearner(trees, 1, 100, 1, 0.0001, subSample, 42, false);

            var indices = Enumerable.Range(0, targets.Length).ToArray();
            indices.Shuffle(new Random(42));
            indices = indices.Take((int)(targets.Length * 0.7)).ToArray();

            var model = sut.Learn(observations, targets, indices);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);
            Console.WriteLine("Error: " + error);
            Console.ReadKey();
        }
    }
}

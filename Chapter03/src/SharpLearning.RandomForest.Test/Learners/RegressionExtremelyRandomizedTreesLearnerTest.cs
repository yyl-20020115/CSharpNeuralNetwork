﻿using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using SharpLearning.RandomForest.Learners;
using SharpLearning.RandomForest.Test.Properties;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SharpLearning.RandomForest.Test.Learners
{
    [TestClass]
    public class RegressionExtremelyRandomizedTreesLearnerTest
    {
        [TestMethod]
        public void RegressionExtremelyRandomizedTreesLearnerTest_Learn_Aptitude_Trees_1()
        {
            var error = RegressionExtremelyRandomizedTreesLearner_Learn_Aptitude(1);
            Assert.AreEqual(0.1278846153846154, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionExtremelyRandomizedTreesLearnerTest_Learn_Aptitude_Trees_5()
        {
            var error = RegressionExtremelyRandomizedTreesLearner_Learn_Aptitude(5);
            Assert.AreEqual(0.098342735042735052, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionExtremelyRandomizedTreesLearnerTest_Learn_Aptitude_Trees_100()
        {
            var error = RegressionExtremelyRandomizedTreesLearner_Learn_Aptitude(100);
            Assert.AreEqual(0.082338163950233079, error, 0.0000001);
        }
        
        [TestMethod]
        public void RegressionExtremelyRandomizedTreesLearnerTest_Learn_Aptitude_Trees_100_SubSample()
        {
            var error = RegressionExtremelyRandomizedTreesLearner_Learn_Aptitude(100, 0.5);
            Assert.AreEqual(0.11122241560374152, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionExtremelyRandomizedTreesLearnerTest_Learn_Glass_1()
        {
            var error = RegressionExtremelyRandomizedTreesLearnerTest_Learn_Glass(1);
            Assert.AreEqual(1.1195386995283647, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionExtremelyRandomizedTreesLearnerTest_Learn_Glass_5()
        {
            var error = RegressionExtremelyRandomizedTreesLearnerTest_Learn_Glass(5);
            Assert.AreEqual(0.36926159796856561, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionExtremelyRandomizedTreesLearnerTest_Learn_Glass_100()
        {
            var error = RegressionExtremelyRandomizedTreesLearnerTest_Learn_Glass(100);
            Assert.AreEqual(0.34674921190387004, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionExtremelyRandomizedTreesLearnerTest_Learn_Glass_100_SubSample()
        {
            var error = RegressionExtremelyRandomizedTreesLearnerTest_Learn_Glass(100, 0.5);
            Assert.AreEqual(0.57461156877304931, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionExtremelyRandomizedTreesLearnerTest_Learn_Glass_100_Indices()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var rows = targets.Length;

            var sut = new RegressionExtremelyRandomizedTreesLearner(100, 1, 100, 1, 0.0001, 1.0, 42, false);
            
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            indices.Shuffle(new Random(42));
            indices = indices.Take((int)(targets.Length * 0.7))
                .ToArray();

            var model = sut.Learn(observations, targets, indices);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.622380056587391, error, 0.0000001);
        }

        double RegressionExtremelyRandomizedTreesLearnerTest_Learn_Glass(int trees, double subSampleRatio = 1.0)
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var rows = targets.Length;

            var sut = new RegressionExtremelyRandomizedTreesLearner(trees, 1, 100, 1, 0.0001, subSampleRatio, 42, false);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);
            return error;
        }

        double RegressionExtremelyRandomizedTreesLearner_Learn_Aptitude(int trees, double subSampleRatio = 1.0)
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var sut = new RegressionExtremelyRandomizedTreesLearner(trees, 1, 100, 1, 0.0001, subSampleRatio, 42, false);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);
            return error;
        }
    }
}

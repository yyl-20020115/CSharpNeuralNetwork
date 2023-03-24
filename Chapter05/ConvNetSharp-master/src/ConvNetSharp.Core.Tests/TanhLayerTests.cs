﻿using System;
using System.Linq;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Core.Tests
{
    [TestClass]
    public class TanhLayerTests
    {
        [TestMethod]
        public void Forward()
        {
            const int inputWidth = 2;
            const int inputHeight = 2;
            const int inputDepth = 2;
            const int inputBatchSize = 2;

            var layer = new TanhLayer<double>();
            layer.Init(inputWidth, inputHeight, inputDepth);

            var input = BuilderInstance.Volume.From(new[]
            {
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
                7.0, 8.0,
                9.0, 10.0,
                11.0, 12.0,
                13.0, 14.0,
                15.0, 16.0
            }, new Shape(inputWidth, inputHeight, inputDepth, inputBatchSize));
            layer.DoForward(input);

            for (int n = 0; n < 2; n++)
            {
                for (int c = 0; c < 2; c++)
                {
                    for (int y = 0; y < 2; y++)
                    {
                        for (int x = 0; x < 2; x++)
                        {
                            Assert.AreEqual(Math.Tanh(input.Get(x, y, c, n)), layer.OutputActivation.Get(x, y, c, n));
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void GradientWrtInputCheck()
        {
            const int inputWidth = 20;
            const int inputHeight = 20;
            const int inputDepth = 2;

            const int batchSize = 3;

            // Create layer
            var layer = new TanhLayer<double>();

            GradientCheckTools.GradientCheck(layer, inputWidth, inputHeight, inputDepth, batchSize, 1e-6);
        }

        [TestMethod]
        public void ComputeTwiceGradientShouldYieldTheSameResult()
        {
            const int inputWidth = 20;
            const int inputHeight = 20;
            const int inputDepth = 2;

            var layer = new TanhLayer<double>();
            layer.Init(inputWidth, inputHeight, inputDepth);

            // Forward pass
            var input = BuilderInstance<double>.Volume.Random(new Shape(inputWidth, inputHeight, inputDepth));
            var output = layer.DoForward(input, true);

            // Set output gradients to 1
            var outputGradient = BuilderInstance<double>.Volume.From(new double[output.Shape.TotalLength].Populate(1.0), output.Shape);

            // Backward pass to retrieve gradients
            layer.Backward(outputGradient);
            var step1 = ((Volume<double>)layer.InputActivationGradients.Clone()).ToArray();

            layer.Backward(outputGradient);
            var step2 = ((Volume<double>)layer.InputActivationGradients.Clone()).ToArray();

            Assert.IsTrue(step1.SequenceEqual(step2));
        }
    }
}
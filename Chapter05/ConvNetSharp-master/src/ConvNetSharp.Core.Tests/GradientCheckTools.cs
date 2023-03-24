﻿using System;
using System.Collections.Generic;
using System.Linq;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Volume;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Core.Tests
{
    public static class GradientCheckTools
    {
        public static void GradientCheck(LayerBase<double> layer, int inputWidth, int inputHeight, int inputDepth, int bactchSize, double epsilon = 1e-4)
        {
            layer.Init(inputWidth, inputHeight, inputDepth);

            // Forward pass
            var input = BuilderInstance<double>.Volume.Random(new Shape(inputWidth, inputHeight, inputDepth, bactchSize), 0.0, Math.Sqrt(2.0 / (inputWidth * inputHeight * inputDepth)));
            var output = layer.DoForward(input, true);

            // Set output gradients to 1
            var outputGradient = BuilderInstance<double>.Volume.From(new double[output.Shape.TotalLength].Populate(1.0), output.Shape);

            // Backward pass to retrieve gradients
            layer.Backward(outputGradient);
            var computedGradients = layer.InputActivationGradients;

            // Now let's approximate gradient using derivate definition
            for (var d = 0; d < inputDepth; d++)
            {
                for (var y = 0; y < inputHeight; y++)
                {
                    for (var x = 0; x < inputWidth; x++)
                    {
                        var oldValue = input.Get(x, y, d);

                        input.Set(x, y, d, oldValue + epsilon);
                        var output1 = (Volume<double>)layer.DoForward(input).Clone();
                        input.Set(x, y, d, oldValue - epsilon);
                        var output2 = (Volume<double>)layer.DoForward(input).Clone();

                        input.Set(x, y, d, oldValue);

                        output1 = output1 - output2;

                        var grad = new double[output.Shape.TotalLength];
                        for (var j = 0; j < output.Shape.TotalLength; j++)
                        {
                            grad[j] = output1.Get(j) / (2.0 * epsilon);
                        }

                        var gradient = grad.Sum(); // approximated gradient
                        var actual = computedGradients.Get(x, y, d);
                        Assert.AreEqual(gradient, actual, 1e-3); // compare layer gradient to the approximated gradient
                    }
                }
            }
        }

        public static void GradienWrtParameterstCheck(int inputWidth, int inputHeight, int inputDepth, int bacthSize, LayerBase<double> layer, double epsilon = 1e-4)
        {
            layer.Init(inputWidth, inputHeight, inputDepth);

            // Forward pass
            var input = BuilderInstance<double>.Volume.From(new double[inputWidth * inputHeight * inputDepth * bacthSize].Populate(1.0), new Shape(inputWidth, inputHeight, inputDepth, bacthSize));
            var output = layer.DoForward(input);

            // Set output gradients to 1
            var outputGradient = BuilderInstance<double>.Volume.From(new double[output.Shape.TotalLength].Populate(1.0), output.Shape);

            // Backward pass to retrieve gradients
            layer.Backward(outputGradient);

            List<ParametersAndGradients<double>> paramsAndGrads = layer.GetParametersAndGradients();

            foreach (var paramAndGrad in paramsAndGrads)
            {
                var vol = paramAndGrad.Volume;
                var gra = paramAndGrad.Gradient;

                // Now let's approximate gradient
                for (var i = 0; i < paramAndGrad.Volume.Shape.TotalLength; i++)
                {
                    input = BuilderInstance<double>.Volume.From(new double[input.Shape.TotalLength].Populate(1.0), input.Shape);

                    var oldValue = vol.Get(i);
                    vol.Set(i, oldValue + epsilon);
                    var output1 = (Volume<double>)layer.DoForward(input).Clone();
                    vol.Set(i, oldValue - epsilon);
                    var output2 = (Volume<double>)layer.DoForward(input).Clone();
                    vol.Set(i, oldValue);

                    output1 = output1 - output2;

                    var grad = new double[output.Shape.TotalLength];
                    for (var j = 0; j < output.Shape.TotalLength; j++)
                    {
                        grad[j] = output1.Get(j) / (2.0 * epsilon);
                    }

                    var gradient = grad.Sum(); // approximated gradient
                    Assert.AreEqual(gradient, gra.Get(i), 1e-3); // compare layer gradient to the approximated gradient
                }
            }
        }
    }
}
﻿using System;
using System.Collections.Generic;
using ConvNetSharp.Core;
using ConvNetSharp.Flow.Layers;
using ConvNetSharp.Flow.Ops;
using ConvNetSharp.Flow.Training;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Tests;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Flow.Tests
{
    [TestClass]
    public abstract class OpTests<T> where T : struct, IEquatable<T>, IFormattable
    {
        [TestMethod]
        public void AddGradientCheck()
        {
            var cns = new ConvNetSharp<T>();
            var shape = new Shape(2, 2, 3, 4);
            var location = NewVolume(RandomUtilities.RandomDoubleArray(shape.TotalLength), shape);

            var x = cns.PlaceHolder("x");
            var z = cns.Const(NewVolume(new double[shape.TotalLength].Populate(1.0), shape), "z");

            GradientCheck(cns, x + z, location);
            GradientCheck(cns, z + x, location);
        }

        [Ignore]
        [TestMethod]
        public void CompareCoreVsFlow()
        {
            var inputWidth = 28;
            var inputHeigth = 28;
            var inputDepth = 3;
            var batchSize = 20;

            #region Flow network

            var netFlow = new Net<T>();
            netFlow.AddLayer(new InputLayer<T>());
            var convLayerFlow1 = new ConvLayer<T>(5, 5, 8) { BiasPref = (T)Convert.ChangeType(0.1, typeof(T)), Stride = 1, Pad = 2 };
            netFlow.AddLayer(convLayerFlow1);
            netFlow.AddLayer(new ReluLayer<T>());
            netFlow.AddLayer(new PoolLayer<T>(2, 2) { Stride = 2 });
            var fullyConnLayerFlow = new FullyConnLayer<T>(10);
            netFlow.AddLayer(fullyConnLayerFlow);
            netFlow.AddLayer(new SoftmaxLayer<T>());

            var trainerFlow = new SgdTrainer<T>(netFlow, (T)Convert.ChangeType(0.01f, typeof(T)))
            {
                BatchSize = batchSize
            };

            #endregion

            #region Core network

            var netCore = new Core.Net<T>();
            netCore.AddLayer(new Core.Layers.InputLayer<T>(inputWidth, inputHeigth, inputDepth));
            var convLayerCore1 = new Core.Layers.ConvLayer<T>(5, 5, 8) { BiasPref = (T)Convert.ChangeType(0.1, typeof(T)), Stride = 1, Pad = 2 };
            netCore.AddLayer(convLayerCore1);
            netCore.AddLayer(new Core.Layers.ReluLayer<T>());
            netCore.AddLayer(new Core.Layers.PoolLayer<T>(2, 2) { Stride = 2 });
            var fullyConnLayerCore = new Core.Layers.FullyConnLayer<T>(10);
            netCore.AddLayer(fullyConnLayerCore);
            netCore.AddLayer(new Core.Layers.SoftmaxLayer<T>(10));

            var trainerCore = new Core.Training.SgdTrainer<T>(netCore)
            {
                LearningRate = (T)Convert.ChangeType(0.01f, typeof(T)),
                BatchSize = batchSize
            };

            #endregion

            // Same weights
            var convfilterCore1 = netFlow.Session.GetVariableByName(netFlow.Op, (convLayerFlow1.Filter as IPersistable<T>).Name);
            convfilterCore1.Result = BuilderInstance<T>.Volume.From(convLayerCore1.Filters.ToArray(), convLayerCore1.Filters.Shape);

            var fullyfilterCore = netFlow.Session.GetVariableByName(netFlow.Op, (fullyConnLayerFlow.Filter as IPersistable<T>).Name);
            fullyfilterCore.Result = BuilderInstance<T>.Volume.From(fullyConnLayerCore.Filters.ToArray(), fullyConnLayerCore.Filters.Shape);

            // Create input
            var xStorage = new double[inputWidth * inputHeigth * inputDepth * batchSize].Populate(1.0);
            var x = NewVolume(xStorage, Volume.Shape.From(inputWidth, inputHeigth, inputDepth, batchSize));

            // Create output
            var yStorage = new double[10 * batchSize];
            var y = NewVolume(yStorage, Volume.Shape.From(1, 1, 10, batchSize));
            for (var i = 0; i < batchSize; i++)
            {
                y.Set(0, 0, i % 10, i, Ops<T>.One);
            }

            for (var k = 0; k < 10; k++)
            {
                xStorage = new double[inputWidth * inputHeigth * inputDepth * batchSize].Populate(1.0 + k);
                x = NewVolume(xStorage, Volume.Shape.From(inputWidth, inputHeigth, inputDepth, batchSize));

                var flowResult = netFlow.Forward(x);
                var coreResult = netCore.Forward(x);

                var sum1 = BuilderInstance<T>.Volume.SameAs(new Shape(1));
                flowResult.DoSum(sum1);
                var sum2 = BuilderInstance<T>.Volume.SameAs(new Shape(1));
                coreResult.DoSum(sum2);
                var diff = Ops<T>.Subtract(sum1.Get(0), sum2.Get(0));

                Console.WriteLine(diff);

                AssertNumber.AreSequenceEqual(flowResult.ToArray(), coreResult.ToArray(), 1e-6);

                trainerCore.Train(x, y);
                trainerFlow.Train(x, y);
            }
        }

        [TestMethod]
        public void DivideGradientCheck()
        {
            var shape = new Shape(2, 2, 3, 4);
            var location = NewVolume(RandomUtilities.RandomDoubleArray(shape.TotalLength), shape);

            var cns = new ConvNetSharp<T>();
            var x = cns.PlaceHolder("x");
            var z = cns.Const(NewVolume(new double[shape.TotalLength].Populate(1.0), shape), "z");

            GradientCheck(cns, x / z, location, 1e-5);
            GradientCheck(cns, z / x, location, 1e-5);
        }

        [TestMethod]
        public void ConcatGradientCheck()
        {
            var leftShape = new Shape(2, 2, 1, 1);
            var rightShape = new Shape(3, 1, 1, 1);

            var cns = new ConvNetSharp<T>();
            var location = NewVolume(RandomUtilities.RandomDoubleArray(leftShape.TotalLength), new Shape(2, 2, 1, 1));
            var x = cns.PlaceHolder("x");
            var z = cns.Const(NewVolume(new double[rightShape.TotalLength].Populate(1.0), rightShape), "z");
            var fun = cns.Concat(x, z);

            GradientCheck(cns, fun, location, 1e-5);
        }

        [TestMethod]
        public void ExpGradientCheck()
        {
            var cns = new ConvNetSharp<T>();
            var x = cns.PlaceHolder("x");
            var fun = cns.Exp(x);

            var shape = new Shape(2, 2, 3, 4);
            var location = NewVolume(RandomUtilities.RandomDoubleArray(shape.TotalLength), shape);

            GradientCheck(cns, fun, location);
        }

        [TestMethod]
        public void FlattenGraddientCheck()
        {
            var cns = new ConvNetSharp<T>();
            var x = cns.PlaceHolder("x");
            var fun = cns.Flatten(x);

            var shape = new Shape(2, 2, 3, 4);
            var location = NewVolume(RandomUtilities.RandomDoubleArray(shape.TotalLength), shape);

            GradientCheck(cns, fun, location);
        }

        private void GradientCheck(ConvNetSharp<T> graph, Op<T> fun, Volume<T> location, double e = 1e-4, Volume<T> outputGrad = null)
        {
            var shape = location.Shape;
            var epsilon = (T)Convert.ChangeType(e, typeof(T));
            var epsilon2 = (T)Convert.ChangeType(2.0 * e, typeof(T));

            if (outputGrad == null)
            {
                outputGrad = NewVolume(new double[shape.TotalLength].Populate(2.0), shape);
            }

            using (var session = new Session<T>())
            {
                var outputGradient = graph.Const(outputGrad, "grad");
                session.Differentiate(fun, outputGradient);

                var dico = new Dictionary<string, Volume<T>> { { "x", location } };
                var x = session.GetVariableByName(fun, "x");

                if (x.Derivate == null)
                {
                    throw new Exception($"Derivate should be defined in {fun.GetType()}");
                }

                var expected = session.Run(x.Derivate, dico);

                var grad = new T[outputGrad.Shape.TotalLength];
                for (var i = 0; i < shape.TotalLength; i++)
                {
                    var old = location.Get(i);

                    location.Set(i, Ops<T>.Add(old, epsilon));
                    var output1 = session.Run(fun, dico).Clone();
                    location.Set(i, Ops<T>.Subtract(old, epsilon));
                    var output2 = session.Run(fun, dico).Clone();

                    location.Set(i, old);

                    output1 = output1 - output2;

                    for (var j = 0; j < outputGrad.Shape.TotalLength; j++)
                    {
                        grad[j] = Ops<T>.Add(Ops<T>.Multiply(Ops<T>.Divide(output1.Get(j), epsilon2), outputGrad.Get(j)), grad[j]);
                    }
                }

                var ratio = outputGrad.Shape.TotalLength / (double)shape.TotalLength;

                for (var i = 0; i < outputGrad.Shape.TotalLength; i++)
                {
                    var gradient = (double)Convert.ChangeType(grad[i], typeof(double)) * ratio;
                    var actual = (double)Convert.ChangeType(expected.Get(i), typeof(double));
                    Assert.IsTrue(Math.Abs(gradient - actual) / (Math.Abs(gradient + actual) + double.Epsilon) < 0.02); // compare layer gradient to the approximated gradient
                }
            }
        }

        [TestMethod]
        public void LogGradientCheck()
        {
            var cns = new ConvNetSharp<T>();
            var x = cns.PlaceHolder("x");
            var fun = cns.Log(x);

            var shape = new Shape(2, 2, 3, 4);
            var location = NewVolume(RandomUtilities.RandomDoubleArray(shape.TotalLength, 20.0, 1.0, true), shape);

            GradientCheck(cns, fun, location, 1e-2);
        }

        [TestMethod]
        public void MultiplyGradientCheck()
        {
            var shape = new Shape(2, 2, 3, 4);
            var location = NewVolume(RandomUtilities.RandomDoubleArray(shape.TotalLength), shape);

            var cns = new ConvNetSharp<T>();
            var x = cns.PlaceHolder("x");
            var z = cns.Const(NewVolume(new double[shape.TotalLength].Populate(2.0), shape), "z");

            GradientCheck(cns, x * z, location);
            GradientCheck(cns, z * x, location);
        }

        protected abstract Volume<T> NewVolume(double[] values, Shape shape);

        [TestMethod]
        public void ReluGradientCheck()
        {
            var cns = new ConvNetSharp<T>();
            var x = cns.PlaceHolder("x");
            var fun = cns.Relu(x);

            var shape = new Shape(2, 2, 3, 4);
            var location = NewVolume(RandomUtilities.RandomDoubleArray(shape.TotalLength), shape);

            GradientCheck(cns, fun, location, 1e-5);
        }

        [TestMethod]
        public void LeakyReluGradientCheck()
        {
            var cns = new ConvNetSharp<T>();
            var x = cns.PlaceHolder("x");
            var fun = cns.LeakyRelu(x, (T)Convert.ChangeType(0.01, typeof(T)));

            var shape = new Shape(2, 2, 3, 4);
            var location = NewVolume(RandomUtilities.RandomDoubleArray(shape.TotalLength), shape);

            GradientCheck(cns, fun, location, 1e-5);
        }

        [TestMethod]
        public void Reshape()
        {
            var cns = new ConvNetSharp<T>();
            var x = cns.PlaceHolder("x");
            var op = cns.Reshape(x, new Shape(1, 1, -1, 1));

            using (var session = new Session<T>())
            {
                // [4] -> [1,1,4,1]
                var result = session.Run(op, new Dictionary<string, Volume<T>> { { "x", NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, Volume.Shape.From(4)) } });
                Assert.AreEqual(new Shape(1, 1, 4, 1), result.Shape);

                // [8] -> [1,1,8,1]
                result = session.Run(op, new Dictionary<string, Volume<T>>
                {
                    {
                        "x", NewVolume(new[]
                        {
                            1.0, 2.0, 3.0, 4.0,
                            1.0, 2.0, 3.0, 4.0
                        }, Volume.Shape.From(8))
                    }
                });
                Assert.AreEqual(new Shape(1, 1, 8, 1), result.Shape);
            }
        }

        [TestMethod]
        public void ReshapeDerivate()
        {
            var cns = new ConvNetSharp<T>();
            var x = cns.PlaceHolder("x");
            var op = cns.Reshape(x, new Shape(1, 1, -1, 1));
            var grad = cns.PlaceHolder("grad");

            using (var session = new Session<T>())
            {
                op.Derivate = grad;
                op.Differentiate();

                var diff = x.Derivate;

                // [4,1,1,1] -> [1,1,4,1]
                var result = session.Run(op, new Dictionary<string, Volume<T>> { { "x", NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, Volume.Shape.From(4, 1, 1, 1)) } });

                // [1,1,4,1] -> [4,1,1,1]
                result = session.Run(diff,
                    new Dictionary<string, Volume<T>>
                    {
                        {"x", NewVolume(new[] {1.0, 2.0, 3.0, 4.0}, Volume.Shape.From(4, 1, 1, 1))},
                        {"grad", NewVolume(new[] {1.0, 1.0, 1.0, 1.0}, Volume.Shape.From(1, 1, 4, 1))}
                    });
                Assert.AreEqual(new Shape(4, 1, 1, 1), result.Shape);
            }
        }

        [TestMethod]
        public void Shape()
        {
            var cns = new ConvNetSharp<T>();
            var x = cns.PlaceHolder("x");
            var op = cns.Shape(x);

            using (var session = new Session<T>())
            {
                // Batch size = 1
                var result = session.Run(op, new Dictionary<string, Volume<T>> { { "x", NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, Volume.Shape.From(1, 1, 4, 1)) } });

                AssertNumber.AreEqual(1.0, result.Get(0));
                AssertNumber.AreEqual(1.0, result.Get(1));
                AssertNumber.AreEqual(4.0, result.Get(2));
                AssertNumber.AreEqual(1.0, result.Get(3));

                // Batch size = 2
                result = session.Run(op, new Dictionary<string, Volume<T>>
                {
                    {
                        "x", NewVolume(new[]
                        {
                            1.0, 2.0, 3.0, 4.0,
                            1.0, 2.0, 3.0, 4.0
                        }, Volume.Shape.From(1, 1, 4, 2))
                    }
                });

                AssertNumber.AreEqual(1.0, result.Get(0));
                AssertNumber.AreEqual(1.0, result.Get(1));
                AssertNumber.AreEqual(4.0, result.Get(2));
                AssertNumber.AreEqual(2.0, result.Get(3));
            }
        }

        [TestMethod]
        public void ShapeIndex()
        {
            var cns = new ConvNetSharp<T>();
            var x = cns.PlaceHolder("x");
            var op = new Shape<T>(cns, x, 2);

            using (var session = new Session<T>())
            {
                // Batch size = 1
                var result = session.Run(op, new Dictionary<string, Volume<T>> { { "x", NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, Volume.Shape.From(1, 1, 4, 1)) } });

                AssertNumber.AreEqual(4.0, result.Get(0));

                // Batch size = 2
                result = session.Run(op, new Dictionary<string, Volume<T>>
                {
                    {
                        "x", NewVolume(new[]
                        {
                            1.0, 2.0, 3.0, 4.0,
                            1.0, 2.0, 3.0, 4.0
                        }, Volume.Shape.From(1, 1, 4, 2))
                    }
                });

                AssertNumber.AreEqual(4.0, result.Get(0));
            }
        }

        [TestMethod]
        public void SigmoidGradientCheck()
        {
            var cns = new ConvNetSharp<T>();
            var x = cns.PlaceHolder("x");
            var fun = cns.Sigmoid(x);

            var shape = new Shape(2, 2, 3, 4);
            var location = NewVolume(RandomUtilities.RandomDoubleArray(shape.TotalLength), shape);

            GradientCheck(cns, fun, location, 1e-3);
        }

        [Ignore]
        [TestMethod]
        public void SoftmaxGradientCheck()
        {
            var cns = new ConvNetSharp<T>();
            var x = cns.PlaceHolder("x");
            var fun = cns.Softmax(x);

            var shape = new Shape(1, 1, 4, 1);
            var location = NewVolume(new[] { 1.0, 0.1, 0.1, 0.1 }, shape);
            var grad = NewVolume(new double[shape.TotalLength], shape);
            grad.Set(0, 0, 1, 0, Ops<T>.One);

            GradientCheck(cns, fun, location, 1e-6, grad);
        }

        [TestMethod]
        public void SqrtGradientCheck()
        {
            var cns = new ConvNetSharp<T>();
            var x = cns.PlaceHolder("x");
            var fun = cns.Sqrt(x);

            var shape = new Shape(2, 2, 3, 4);
            var location = NewVolume(RandomUtilities.RandomDoubleArray(shape.TotalLength, posisitveOnly: true), shape);

            GradientCheck(cns, fun, location);
        }

        [TestMethod]
        public void PowerGradientCheck()
        {
            var cns = new ConvNetSharp<T>();
            var shape = new Shape(2, 2, 3, 4);
            var location = NewVolume(RandomUtilities.RandomDoubleArray(shape.TotalLength), shape);

            var x = cns.PlaceHolder("x");
            var z = cns.Const(NewVolume(new double[shape.TotalLength].Populate(2.0), shape), "z");

            GradientCheck(cns, x ^ z, location);
            GradientCheck(cns, z ^ x, location);
        }

        [TestMethod]
        public void SubstractGradientCheck()
        {
            var shape = new Shape(2, 2, 3, 4);
            var location = NewVolume(RandomUtilities.RandomDoubleArray(shape.TotalLength), shape);

            var cns = new ConvNetSharp<T>();
            var x = cns.PlaceHolder("x");
            var z = cns.Const(NewVolume(new double[shape.TotalLength].Populate(1.0), shape), "z");

            GradientCheck(cns, x - z, location);
            GradientCheck(cns, z - x, location);
        }

        [TestMethod]
        public void SumGradientCheck()
        {
            var cns = new ConvNetSharp<T>();
            var x = cns.PlaceHolder("x");
            var fun = cns.Sum(x, new Shape(1, 1, 1, 4));

            var shape = new Shape(2, 2, 3, 4);
            var location = NewVolume(RandomUtilities.RandomDoubleArray(shape.TotalLength), shape);

            var grad = NewVolume(new[] { 1.0, 1.0, 1.0, 1.0 }, new Shape(1, 1, 1, 4));
            GradientCheck(cns, fun, location, 1e-4, grad);
        }

        [TestMethod]
        public void SumOp()
        {
            var cns = new ConvNetSharp<T>();
            var x = cns.Const(NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3)), "x");
            var op = cns.Sum(x, new Shape(1));

            using (var session = new Session<T>())
            {
                var result = op.Evaluate(session);
                AssertNumber.AreEqual(6.0, result.Get(0));
            }
        }

        [TestMethod]
        public void SumOpBatch()
        {
            var cns = new ConvNetSharp<T>();
            var x = cns.Const(NewVolume(new[]
            {
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
            }, new Shape(3, 1, 1, 2)), "x");
            var op = cns.Sum(x, new Shape(1, 1, 1, 2));

            using (var session = new Session<T>())
            {
                var result = op.Evaluate(session);
                AssertNumber.AreEqual(6.0, result.Get(0));
                AssertNumber.AreEqual(15.0, result.Get(1));
            }
        }

        [TestMethod]
        public void SumOpDerivative()
        {
            var cns = new ConvNetSharp<T>();
            var x = cns.Const(NewVolume(new[] { 1.0, 2.0, 3.0 }, new Shape(3)), "x");
            var op = cns.Sum(x, new Shape(1));

            using (var session = new Session<T>())
            {
                session.Differentiate(op);

                op.Derivate = cns.Const(NewVolume(new double[1].Populate(50.0), new Shape(1)), "50");

                var result = x.Derivate.Evaluate(session);
                Assert.AreEqual(result.Shape, new Shape(3));
            }
        }

        [TestMethod]
        public void ConcatOpDerivative()
        {
            var cns = new ConvNetSharp<T>();

            var leftShape = new Shape(2, 2);
            var rightShape = new Shape(2);

            var left = cns.Const(NewVolume(new[] { 1.0, 2.0, 3.0, 4.0 }, leftShape), "left");
            var right = cns.Const(NewVolume(new[] { 4.0, 5.0 }, rightShape), "right");

            var op = cns.Concat(left, right);

            using (var session = new Session<T>())
            {
                session.Differentiate(op);

                var total = (int)(leftShape.TotalLength + rightShape.TotalLength);
                op.Derivate = cns.Const(NewVolume(new double[total].Populate(1.0), new Shape(total)), "Gradient");

                var result = left.Derivate.Evaluate(session);
                Assert.AreEqual(result.Shape, leftShape);

                result = right.Derivate.Evaluate(session);
                Assert.AreEqual(result.Shape, rightShape);
            }
        }

        [TestMethod]
        public void TanhGradientCheck()
        {
            var cns = new ConvNetSharp<T>();
            var x = cns.PlaceHolder("x");
            var fun = cns.Tanh(x);

            var shape = new Shape(2, 2, 3, 4);
            var location = NewVolume(RandomUtilities.RandomDoubleArray(shape.TotalLength), shape);

            GradientCheck(cns, fun, location);
        }
    }
}
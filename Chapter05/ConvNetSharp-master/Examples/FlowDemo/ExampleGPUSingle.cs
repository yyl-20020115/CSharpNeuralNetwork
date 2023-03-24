﻿using System;
using System.Collections.Generic;
using ConvNetSharp.Flow;
using ConvNetSharp.Flow.Training;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.GPU.Single;

namespace FlowDemo
{
    public class ExampleGpuSingle
    {
        /// <summary>
        /// Solves y = x * W + b (GPU version)
        /// for y = 1 and x = -2
        /// </summary>
        public static void Example1()
        {
            var cns = new ConvNetSharp<float>();

            BuilderInstance<float>.Volume = new VolumeBuilder();

            // Graph creation
            var x = cns.PlaceHolder("x");
            var y = cns.PlaceHolder("y");

            var W = cns.Variable(1.0f, "W", true);
            var b = cns.Variable(2.0f, "b", true);

            var fun = x * W + b;

            var cost = (fun - y) * (fun - y);

            var optimizer = new GradientDescentOptimizer<float>(cns, learningRate: 0.01f);

            using (var session = new Session<float>())
            {
                session.Differentiate(cost); // computes dCost/dW at every node of the graph

                double currentCost;
                do
                {
                    var dico = new Dictionary<string, Volume<float>> { { "x", -2.0f }, { "y", 1.0f } };

                    currentCost = session.Run(cost, dico);
                    Console.WriteLine($"cost: {currentCost}");

                    var result = session.Run(fun, dico);
                    session.Run(optimizer, dico);
                } while (currentCost > 1e-5);

                double finalW = W.Result;
                double finalb = b.Result;
                Console.WriteLine($"fun = x * {finalW} + {finalb}");
                Console.ReadKey();
            }
        }

        /// <summary>
        /// Computes and displays t = t + 1
        /// </summary>
        public static void Example3()
        {
            var cns = new ConvNetSharp<float>();
            BuilderInstance<float>.Volume = new VolumeBuilder();

            // Graph creation
            var t = cns.Variable(0.0f, "t", true);
            var fun = cns.Assign(t, t + 1.0f);

            using (var session = new Session<float>())
            {
                session.InitializePlaceHolders(fun, new Dictionary<string, Volume<float>> { { "t", 1.0f } });

                do
                {
                    session.Run(fun, null);

                    var x = t.Result.Get(0);
                    Console.WriteLine(x);

                } while (!Console.KeyAvailable);
            }
        }
    }
}
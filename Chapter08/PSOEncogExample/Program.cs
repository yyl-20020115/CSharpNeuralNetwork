using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PSOEncogExample
{
    using Encog;
    using Encog.MathUtil.Randomize;
    using Encog.ML.Data;
    using Encog.ML.Data.Basic;
    using Encog.ML.Train;
    using Encog.Neural.Networks;
    using Encog.Neural.Networks.Training;
    using Encog.Neural.Networks.Training.PSO;
    using Encog.Util.Simple;

    class Program
    {
        /// <summary>
        /// Input for the XOR function.
        /// </summary>
        public static double[][] XORInput = {
            new[] {0.0, 0.0},
            new[] {1.0, 0.0},
            new[] {0.0, 1.0},
            new[] {1.0, 1.0}
        };

        /// <summary>
        /// Ideal output for the XOR function.
        /// </summary>
        public static double[][] XORIdeal = {
            new[] {0.0},
            new[] {1.0},
            new[] {1.0},
            new[] {0.0}
        };

        static void Main(string[] args)
        {
            Console.ReadLine();
            IMLDataSet trainingSet = new BasicMLDataSet(XORInput, XORIdeal);
            BasicNetwork network = EncogUtility.SimpleFeedForward(2, 2, 0, 1, false);
            ICalculateScore score = new TrainingSetScore(trainingSet);
            IRandomizer randomizer = new NguyenWidrowRandomizer();

            IMLTrain train = new NeuralPSO(network, randomizer, score, 40);

            EncogUtility.TrainToError(train, 0.01);

            network = (BasicNetwork)train.Method;

            // test the neural network
            Console.WriteLine("Neural Network Results:");
            EncogUtility.Evaluate(network, trainingSet);

            Console.ReadLine();
            EncogFramework.Instance.Shutdown();
        }
    }
}

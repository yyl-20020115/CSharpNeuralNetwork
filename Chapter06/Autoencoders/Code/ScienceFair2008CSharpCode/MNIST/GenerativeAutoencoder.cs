using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork;
using System.IO;

namespace ScienceFair2008
{
    public class GenerativeAutoencoder: MNISTProgramBase
    {
        private int[] layersizes = null;

        public int[] LayerSizes
        {
            get { return layersizes; }
            set { layersizes = value; }
        }

        public override void Init()
        {
            InitInputProvider();
            LoadNetwork();
            SetNetLearnrates(0.001, 0.1);
        }

        private void LoadNetwork()
        {
            if (loadnetname != null)
            {
                net = Network.Load(CreateSpecificFilePath(loadnetname));
            }
            else
            {
                net = CreatePreTrainedNet();
            }
        }

        #region PreTraining
        private Network CreatePreTrainedNet()
        {
            Network pretrainnet = InitPreTrainedNet();
            PreTrain(pretrainnet, 60000);
            return PreTrainComplete(pretrainnet);
        }
        private Network InitPreTrainedNet()
        {
            int numlayers = layersizes.GetLength(0);
            ILayer[] layers = new ILayer[numlayers];
            double[] learnrates = new double[4];
            for (int i = 0; i < numlayers; i++)
            {
                layers[i] = new SigmoidLayer(layersizes[i], 0.01);
                learnrates[i] = 0.01;
            }
            return new Network(new SigmoidLayer(784, 0.01), layers, learnrates);
        }
        private void PreTrain(Network PPreTrainNet, int PNumCases)
        {
            for (int i = 0; i < PPreTrainNet.NumLayers; i++)
            {
                for (int j = 0; j < PNumCases; j++)
                {
                    PPreTrainNet.PreTrain(inputprovider, i);
                    Console.WriteLine("      " + j);
                }
            }
        }
        private Network PreTrainComplete(Network PPreTrainNet)
        {
            int numlayerspt = PPreTrainNet.NumLayers;
            int numlayersnew = numlayerspt * 2;
            ILayer inputlayer = PPreTrainNet.InputLayer.Clone();
            ILayer[] layers = new ILayer[numlayersnew];
            double[] learnrate = new double[numlayersnew];
            double[][][] weights = new double[numlayersnew][][];
            for (int i = 0; i < numlayerspt; i++)
            {
                ILayer copylayer = PPreTrainNet.Layers[i];
                layers[i] = copylayer.Clone();
                learnrate[i] = PPreTrainNet.Learnrates[i];
                weights[i] = (double[][])PPreTrainNet.Weights[i].Clone();
            }
            for (int i = 0; i < numlayerspt - 1; i++)
            {
                ILayer copylayer = PPreTrainNet.Layers[i];
                int copydestination = numlayerspt * 2 - i - 2;
                layers[copydestination] = copylayer.Clone();
                learnrate[copydestination] = PPreTrainNet.Learnrates[i];
            }
            for (int i = 0; i < numlayerspt; i++)
            {
                int copydestination = numlayerspt * 2 - i - 1;
                weights[copydestination] = Utility.TransposeArray(PPreTrainNet.Weights[i]);
            }
            layers[numlayersnew - 1] = PPreTrainNet.InputLayer.Clone();
            learnrate[numlayersnew - 1] = PPreTrainNet.Learnrates[0];
            weights[numlayersnew - 1] = Utility.TransposeArray(PPreTrainNet.Weights[0]);
            return new Network(inputlayer, layers, weights, null, learnrate);
        }
        #endregion

        private void InitInputProvider()
        {
            inputprovider = new GenerativeInputProvider(CreateDataFilePath(trainingset), CreateDataFilePath(testingset),
                                                   CreateDataFilePath(traininglabels), CreateDataFilePath(testinglabels),
                                                   numtrainingcases, numtestingcases, 
                                                   imagesize);
        }

        public override void Display()
        {
            InitTotem();
            Core.Run();
        }
        protected override double[] SetCurrentInput()
        {
            return net.InputLayer.Outputs;
        }
        protected override double[] SetCurrentOutputs()
        {
            net.Run(inputprovider);
            return net.Layers[net.NumLayers - 1].Outputs;
        }

        public override string CreateSpecificFilePath(string PFileName)
        {
            return ("Data\\GenerativeAutoencoder\\" + PFileName);
        }
    }
}

using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork;
using System.IO;

namespace ScienceFair2008
{
    public class AutoencoderBasedDiscrimination:DiscriminativeProgramBase
    {
        private string basenetname = null;
        public string BaseNetName
        {
            get { return basenetname; }
            set { basenetname = value; }
        }

        public override void Init()
        {
            InitInputProvider();
            LoadNetwork();
            SetNetLearnrates(0.005, 0.1);
        }

        private void LoadNetwork()
        {
            if (loadnetname != null)
            {
                net = Network.Load(CreateSpecificFilePath(loadnetname));
            }
            else
            {
                net = CreateNetwork(CreateSpecificFilePath(basenetname));
            }
        }

        private Network CreateNetwork(string PBaseNetName)
        {
            if (PBaseNetName == null)
            {
                throw new Exception("AAARGGHH!");
            }
            Network tempacnet = Network.Load(PBaseNetName);
            int numlayersnew = tempacnet.NumLayers / 2 + 1;
            ILayer[] layers = new ILayer[numlayersnew];
            double[][][] weights = new double[numlayersnew][][];
            double[] learnrates = new double[numlayersnew];
            for (int i = 0; i < numlayersnew; i++)
            {
                layers[i] = tempacnet.Layers[i];
                weights[i] = tempacnet.Weights[i];
                learnrates[i] = tempacnet.Learnrates[i];
            }
            layers[numlayersnew - 1] = new SigmoidLayer(10, 0.001);
            weights[numlayersnew - 1] = new double[10][];
            for(int i = 0;i < 10;i++)
            {
                int numneurons = layers[numlayersnew - 2].NumNeurons;
                weights[numlayersnew - 1][i] = new double[numneurons];
                for(int j = 0;j < numneurons;j++)
                {
                    weights[numlayersnew - 1][i][j] = 0;
                }
            }
            learnrates[numlayersnew - 1] = 0.001;
            return new Network(tempacnet.InputLayer, layers, weights, null, learnrates);
        }

        public override void Display()
        {
            InitTotem();
            Core.Run();
        }

        public override string CreateSpecificFilePath(string PFileName)
        {
            return ("Data\\AutoencoderBasedDiscrimination\\" + PFileName);
        }
    }
}

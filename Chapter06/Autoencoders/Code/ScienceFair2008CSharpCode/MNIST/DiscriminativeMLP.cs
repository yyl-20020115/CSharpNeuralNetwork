using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork;
using System.IO;

namespace ScienceFair2008
{
    public class DiscriminativeMLP: DiscriminativeProgramBase
    {
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
                net = CreateNetwork();
            }
        }

        private Network CreateNetwork()
        {
            ILayer[] layers = new ILayer[2];
            layers[0] = new SigmoidLayer(300, 0.001);
            layers[1] = new SigmoidLayer(10, 0.001);
            double[] learnrates = new double[2];
            learnrates[0] = 0.001;
            learnrates[1] = 0.001;
            return new Network(new SigmoidLayer(784, 0.001), layers, learnrates);
        }

        public override string CreateSpecificFilePath(string PFileName)
        {
            return ("Data\\DiscriminativeMLP\\" + PFileName);
        }
    }
}

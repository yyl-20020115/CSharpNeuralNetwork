using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork;

namespace ScienceFair2008
{
    public class AutoencoderToMLP : DiscriminativeProgramBase
    {
        private int autoencoderoutputs = 30;
        private string autoencoderfilename = null;

        public int AutoencoderOutputs
        {
            get { return autoencoderoutputs; }
            set { autoencoderoutputs = value; }
        }
        public string AutoencoderFilename
        {
            get { return autoencoderfilename; }
            set { autoencoderfilename = value; }
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
                net = CreateNetwork();
            }
        }
        private Network CreateNetwork()
        {
            ILayer[] layers = new ILayer[2];
            layers[0] = new SigmoidLayer(300, 0.005);
            layers[1] = new SigmoidLayer(10, 0.005);
            double[] learnrates = new double[2];
            learnrates[0] = 0.005;
            learnrates[1] = 0.005;
            return new Network(new SigmoidLayer(autoencoderoutputs, 0.005), layers, learnrates);
        }
        protected override void InitInputProvider()
        {
            inputprovider = new AutoencoderInputProvider(CreateDataFilePath(trainingset), CreateDataFilePath(testingset),
                                                   CreateDataFilePath(traininglabels), CreateDataFilePath(testinglabels),
                                                   numtrainingcases, numtestingcases,
                                                   imagesize, numoutputs, 
                                                   CreateSpecificFilePath(autoencoderfilename));
        }

        public override void Display()
        {
            InitTotem();
            Core.Run();
        }

        public override string CreateSpecificFilePath(string PFileName)
        {
            return ("Data\\AutoencoderToMLP\\" + PFileName);
        }
    }
}

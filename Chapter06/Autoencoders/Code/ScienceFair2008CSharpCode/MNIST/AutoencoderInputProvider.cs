using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork;

namespace ScienceFair2008
{
    public class AutoencoderInputProvider: MNISTInputProvider
    {
        private int numoutputs;
        private Network autoencoder;
        public AutoencoderInputProvider(string PTrainingFileName, string PTestingFileName,
                                  string PTrainingLabelFile, string PTestingLabelFile,
                                  int PTrainingSetSize, int PTestingSetSize,
                                  int PImageSize, int PNumOutputs, string PAutoencoderFile)
            :base(PTrainingFileName, PTestingFileName,
                      PTrainingLabelFile, PTestingLabelFile,
                      PTrainingSetSize, PTestingSetSize,
                      PImageSize)
        {
            numoutputs = PNumOutputs;
            autoencoder = Network.Load(PAutoencoderFile);
        }
        public override double[] DesiredOutput()
        {
            double[] retval;
            Utility.InitDefaultArray(out retval, numoutputs, 0);
            if (testing)
            {
                retval[testinglabels[testingindex - 1]] = 1;
            }
            else
            {
                retval[traininglabels[trainingindex - 1]] = 1;
            }
            return retval;
        }
        public override double[] NextInputCase()
        {
            double[] data = base.NextInputCase();
            return autoencoder.GetLayerOutput((autoencoder.NumLayers / 2) - 1, data);
        }
    }
}

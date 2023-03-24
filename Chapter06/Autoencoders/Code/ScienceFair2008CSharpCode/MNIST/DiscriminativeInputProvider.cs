using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork;

namespace ScienceFair2008
{
    public class DiscriminativeInputProvider:MNISTInputProvider
    {
        private int numoutputs;
        public DiscriminativeInputProvider(string PTrainingFileName, string PTestingFileName,
                                  string PTrainingLabelFile, string PTestingLabelFile,
                                  int PTrainingSetSize, int PTestingSetSize,
                                  int PImageSize, int PNumOutputs)
            :base(PTrainingFileName, PTestingFileName,
                      PTrainingLabelFile, PTestingLabelFile,
                      PTrainingSetSize, PTestingSetSize,
                      PImageSize)
        {
            numoutputs = PNumOutputs;
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
    }
}

using System;
using System.Collections.Generic;
using System.Text;

namespace ScienceFair2008
{
    public class GenerativeInputProvider: MNISTInputProvider
    {
        public GenerativeInputProvider(string PTrainingFileName, string PTestingFileName,
                                  string PTrainingLabelFile, string PTestingLabelFile,
                                  int PTrainingSetSize, int PTestingSetSize,
                                  int PImageSize)
            :base(PTrainingFileName, PTestingFileName,
                      PTrainingLabelFile, PTestingLabelFile,
                      PTrainingSetSize, PTestingSetSize,
                      PImageSize)
        {
        }
        public override double[] DesiredOutput()
        {
            return curdata;
        }
    }
}

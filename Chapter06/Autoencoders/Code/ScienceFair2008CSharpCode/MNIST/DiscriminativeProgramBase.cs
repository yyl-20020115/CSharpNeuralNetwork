using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace ScienceFair2008
{
    public abstract class DiscriminativeProgramBase: MNISTProgramBase
    {
        protected int numoutputs = 10;

        public int NumOutputs
        {
            get { return numoutputs; }
            set { numoutputs = value; }
        }

        protected virtual void InitInputProvider()
        {
            inputprovider = new DiscriminativeInputProvider(CreateDataFilePath(trainingset), CreateDataFilePath(testingset),
                                                   CreateDataFilePath(traininglabels), CreateDataFilePath(testinglabels),
                                                   numtrainingcases, numtestingcases,
                                                   imagesize, numoutputs);
        }

        protected override double Train(int PNumCases)
        {
            CalculateHessian();
            double mse = 0;
            int numcorrect = 0;
            double percent = 0;
            for (int i = 0; i < PNumCases; i++)
            {
                net.Train(inputprovider);
                double[] desiredoutput = inputprovider.DesiredOutput();
                mse += net.CalculateMSE(net.NumLayers - 1, desiredoutput);
                Console.WriteLine(mse / (i + 1) + "     " + i);
                if (CheckOutputCorrect(desiredoutput, net.GetOutput()))
                {
                    numcorrect++;
                }
                percent = ((double)numcorrect) / ((double)i + 1) * 100;
                Console.WriteLine("Percentage Correct: " + percent);
            }
            WriteOneLineFile(saveprefix + "trainingpercent.txt", percent);
            return mse / PNumCases;
        }
        protected override double Test(int PNumCases)
        {
            inputprovider.Testing = true;
            double mse = 0;
            int numcorrect = 0;
            double percent = 0;
            for (int i = 0; i < PNumCases; i++)
            {
                net.Run(inputprovider);
                double[] desiredoutput = inputprovider.DesiredOutput();
                mse += net.CalculateMSE(net.NumLayers - 1, desiredoutput);
                Console.WriteLine(mse / (i + 1) + "     " + i);
                if (CheckOutputCorrect(desiredoutput, net.GetOutput()))
                {
                    numcorrect++;
                }
                percent = ((double)numcorrect) / ((double)i + 1) * 100;
                Console.WriteLine("Percentage Correct: " + percent);
            }
            WriteOneLineFile(saveprefix + "testingpercent.txt", percent);
            inputprovider.Testing = false;
            return mse / PNumCases;
        }
        private bool CheckOutputCorrect(double[] PDesiredOutput, double[] PRealOutput)
        {
            int highest = 0;
            double highestval = PRealOutput[0];
            for (int i = 1; i < numoutputs; i++)
            {
                double output = PRealOutput[i];
                if (output > highestval)
                {
                    highest = i;
                    highestval = output;
                }
            }
            if (PDesiredOutput[highest] == 1)
            {
                return true;
            }
            return false;
        }
    }
}

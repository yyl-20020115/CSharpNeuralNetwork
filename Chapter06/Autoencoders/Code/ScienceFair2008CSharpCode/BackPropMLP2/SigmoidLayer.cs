using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork;
using System.IO;

namespace NeuralNetwork
{
    public class SigmoidLayer : ILayer
    {
        public SigmoidLayer(int PSize, double PBiasLearnrate)
                :base(PSize, PBiasLearnrate)
        {}
        internal SigmoidLayer(BinaryReader PFile)
                :base(PFile)
        {}
        public override void Process(int PWhich, double PInput)
        {
            output[PWhich] = ProcessNoSet(PWhich, PInput);
        }

        public override double FirstDerivative(double POutput)
        {
            return (POutput * (1 - POutput));
        }

        public override double ProcessNoSet(int PWhich, double PInput)
        {
            double input = PInput + bias[PWhich];
            double retval = 0;
            if (input < -3)
            {
                retval = 0;
            }
            else if (input > 3)
            {
                retval = 1;
            }
            else
            {
                retval = (1 / (1 + Math.Exp(-input)));
            }
            return retval;
        }

        public override ILayer Clone()
        {
            SigmoidLayer retval = new SigmoidLayer(numneurons, biaslearnrate);
            retval.bias = (double[])bias.Clone();
            retval.output = (double[])output.Clone();
            return retval;
        }

        public override void Write(System.IO.BinaryWriter PFile)
        {
            PFile.Write(0);
            PFile.Write(numneurons);
            PFile.Write(biaslearnrate);
            for (int i = 0; i < numneurons; i++)
            {
                PFile.Write(bias[i]);
            }
        }
    }
}

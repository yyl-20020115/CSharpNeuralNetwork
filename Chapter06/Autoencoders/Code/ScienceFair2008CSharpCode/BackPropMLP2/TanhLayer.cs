using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace NeuralNetwork
{
    public class TanhLayer: ILayer
    {
        public TanhLayer(int PSize, double PBiasLearnrate)
                :base(PSize, PBiasLearnrate)
        {}
        internal TanhLayer(BinaryReader PFile)
                :base(PFile)
        {}
        public override void Process(int PWhich, double PInput)
        {
            output[PWhich] = ProcessNoSet(PWhich, PInput);
        }

        private double CalculateTanh(double PInput)
        {
            if (PInput < -3)
            {
                 return -1;
            }
            else if (PInput > 3)
            {
                 return 1;
            }
            else
            {
                 return Math.Tanh(PInput);
            }
        }

        public override double ProcessNoSet(int PWhich, double PInput)
        {
            double input = PInput + bias[PWhich];
            return CalculateTanh(input);
        }

        public override double FirstDerivative(double POutput)
        {
            return 1 - (POutput * POutput);
        }

        public override ILayer Clone()
        {
            TanhLayer retval = new TanhLayer(numneurons, biaslearnrate);
            retval.bias = (double[])bias.Clone();
            retval.output = (double[])output.Clone();
            return retval;
        }

        public override void Write(System.IO.BinaryWriter PFile)
        {
            PFile.Write(1);
            PFile.Write(numneurons);
            PFile.Write(biaslearnrate);
            for (int i = 0; i < numneurons; i++)
            {
                PFile.Write(bias[i]);
            }
        }
    }
}

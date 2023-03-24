using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace NeuralNetwork
{
    public abstract class ILayer
    {
        #region Vars
        protected double[] output;
        protected double[] bias;
        protected double biaslearnrate;
        protected int numneurons = 0;
        #endregion

        #region Accessors
        public double[] Outputs
        {
            get
            {
                return output;
            }
        }
        public double[] Biases
        {
            get
            {
                return bias;
            }
            set
            {
                bias = value;
            }
        }
        public double BiasLearnrate
        {
            get
            {
                return biaslearnrate;
            }
        }
        public int NumNeurons
        {
            get
            {
                return numneurons;
            }
        }
        #endregion


        public ILayer()
        {

        }
        public ILayer(int PSize, double PBiasLearnrate)
        {
            if (PSize <= 0)
            {
                throw new Exception("Can't have a layer without neurons!");
            }
            numneurons = PSize;
            output = new double[numneurons];
            bias = new double[numneurons];
            for (int i = 0; i < numneurons; i++)
            {
                output[i] = 0;
                bias[i] = 0;
            }
            biaslearnrate = PBiasLearnrate;
        }
        protected ILayer(BinaryReader PFile)
        {
            numneurons = PFile.ReadInt32();
            biaslearnrate = PFile.ReadDouble();
            output = new double[numneurons];
            bias = new double[numneurons];
            for (int i = 0; i < numneurons; i++)
            {
                bias[i] = PFile.ReadDouble();
                output[i] = 0;
            }
        }

        public abstract double ProcessNoSet(int PWhich, double PInput);
        public abstract void Process(int PWhich, double PInput);
        public abstract double FirstDerivative(double POutput);
        public abstract ILayer Clone();

        public abstract void Write(System.IO.BinaryWriter PFile);

        public static ILayer Load(System.IO.BinaryReader PFile)
        {
            int type = PFile.ReadInt32();
            if (type == 0)
            {
                return new SigmoidLayer(PFile);
            }
            else if (type == 0)
            {
                return new TanhLayer(PFile);
            }
            return null;
        }
    }
}

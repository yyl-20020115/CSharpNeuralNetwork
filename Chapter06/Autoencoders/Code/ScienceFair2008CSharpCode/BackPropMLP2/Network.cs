using System;
using System.Collections.Generic;
using System.Text;
using NNBase;
using System.IO;

namespace NeuralNetwork
{
    enum TDerivative
    {
        First,
        Second
    }
    public class Network:NNBaseClass
    {
        private static Random rand = new Random();
        private int numlayers;
        private int numinputs;
        private ILayer inputlayer;
        private ILayer[] layers;
        private double[][][] weights;
        private double[] learnrates;
        private double[][][] diagonalhessian;
        private double blowuppreventer = 0.1;

        #region Accessors
        public int NumLayers
        {
            get { return numlayers; }
        }
        public int NumInputs
        {
            get { return numinputs; }
        }
        public ILayer InputLayer
        {
            get { return inputlayer; }
        }
        public ILayer[] Layers
        {
            get { return layers; }
        }
        public double[][][] Weights
        {
            get { return weights; }
        }
        public double[][] Biases
        {
            get
            {
                double[][] retval = new double[numlayers + 1][];
                retval[0] = inputlayer.Biases;
                for (int i = 0; i < numlayers; i++)
                {
                    retval[i + 1] = layers[i].Biases;
                }
                return retval;
            }
        }
        public double[] Learnrates
        {
            get { return learnrates; }
            set { learnrates = value; }
        }
        public double BlowUpPreventer
        {
            get
            {
                return blowuppreventer;
            }
            set
            {
                blowuppreventer = value;
            }
        }
        #endregion

        #region Initialization
        private Network()
        { }
        public Network(ILayer PInputLayer, ILayer[] PLayers,
                       double[][][] PInitialWeights, double[][] PInitialBiases,
                       double[] PLearnrates)
        {
            //TODO: Error Checking!!
            numinputs = PInputLayer.NumNeurons;
            numlayers = PLayers.GetLength(0);
            layers = PLayers;
            inputlayer = PInputLayer;
            learnrates = PLearnrates;
            InitWeights(PInputLayer.NumNeurons, PInitialWeights);
            if(PInitialBiases != null)
            {
                InitBiases(PInitialBiases);
            }
        }

        private void InitBiases(double[][] PInitialBiases)
        {
            for (int i = 0; i < numlayers; i++)
            {
                layers[i].Biases = (double[])PInitialBiases[i].Clone();
            }
        }

        private void InitWeights(int PNumInputs, double[][][] PInitialWeights)
        {
            weights = new double[numlayers][][];
            for (int i = 0; i < numlayers; i++)
            {
                int numneurons = layers[i].NumNeurons;
                weights[i] = new double[numneurons][];
                for (int j = 0; j < numneurons; j++)
                {
                    int numbackweights = PNumInputs;
                    if (i != 0)
                    {
                        numbackweights = layers[i - 1].NumNeurons;
                    }
                    weights[i][j] = new double[numbackweights];
                    for (int k = 0; k < numbackweights; k++)
                    {

                        weights[i][j][k] = rand.NextDouble() / 10;
                        if (PInitialWeights != null)
                        {
                            weights[i][j][k] = PInitialWeights[i][j][k];
                        }
                    }
                }
            }
        }

        public Network(ILayer PInputLayer, ILayer[] PLayers,
                       double[] PLearnrates)
            :this(PInputLayer, PLayers, null, null, PLearnrates)
        {

        }
        #endregion

        public double CalculateMSE(int PLayer, double[] PDesired)
        {
            ILayer whichlayer = GetLayer(PLayer);
            int numneurons = whichlayer.NumNeurons;
            double error = 0;
            double[] outputs = whichlayer.Outputs;
            for (int i = 0; i < numneurons; i++)
            {
                error += (PDesired[i] - outputs[i]) * (PDesired[i] - outputs[i]);
            }
            //error /= ((double)numneurons);
            return error;
        }

        #region PreTrain
        public void PreTrain(INNInputProvider PProvider, int PWhichWeightSet)
        {
            double[] input = PProvider.NextInputCase();
            double[] posvis;
            double[] poshid;
            double[] negvis;
            double[] neghid;
            GetPreTrainData(PWhichWeightSet, input, out posvis, out poshid, out negvis, out neghid);
            PerformPreTraining(PWhichWeightSet, posvis, poshid, negvis, neghid);
        }

        private void GetPreTrainData(int PWhichWeightSet, double[] PInput,
                                     out double[] PPosVis, out double[] PPosHid,
                                     out double[] PNegVis, out double[] PNegHid)
        {
            SetInput(PInput);
            for (int i = 0; i < PWhichWeightSet; i++)
            {
                ProcessLayerForwards(i, GetLayer(i - 1).Outputs);
            }
            PPosVis = ((double[])GetLayer(PWhichWeightSet - 1).Outputs.Clone());

            ProcessLayerForwards(PWhichWeightSet, PPosVis);
            PPosHid = (double[])GetLayer(PWhichWeightSet).Outputs.Clone();

            ProcessLayerBackwards(PWhichWeightSet - 1, PPosHid);
            PNegVis = (double[])GetLayer(PWhichWeightSet - 1).Outputs.Clone();

            ProcessLayerForwards(PWhichWeightSet, PNegVis);
            PNegHid = (double[])GetLayer(PWhichWeightSet).Outputs.Clone();
            Console.Write(CalculateMSE(PWhichWeightSet - 1, PPosVis));
        }

        private void PerformPreTraining(int PWhichWeightSet, double[] PPosVis,
                                        double[] PPosHid, double[] PNegVis,
                                        double[] PNegHid)
        {
            int numvisibles = PPosVis.GetLength(0);
            int numhiddens = PPosHid.GetLength(0);
            double[][] trainweightset = weights[PWhichWeightSet];
            double[] visiblebias = GetLayer(PWhichWeightSet - 1).Biases;
            double[] hiddenbias = GetLayer(PWhichWeightSet).Biases;
            double weightlearnrate = learnrates[PWhichWeightSet];
            double biaslearnratevisible = GetLayer(PWhichWeightSet - 1).BiasLearnrate;
            double biaslearnratehidden = GetLayer(PWhichWeightSet).BiasLearnrate;
            for (int i = 0; i < numvisibles; i++)
            {
                for (int j = 0; j < numhiddens; j++)
                {
                    double weighttrainamnt = (PPosVis[i] * PPosHid[j]) - (PNegVis[i] * PNegHid[j]);
                    trainweightset[j][i] += weightlearnrate * weighttrainamnt;
                }
                double biastrainamnt = PPosVis[i] - PNegVis[i];
                visiblebias[i] += biaslearnratevisible * biastrainamnt;
            }
            for (int i = 0; i < numhiddens; i++)
            {
                double biastrainamnt = PPosHid[i] - PNegHid[i];
                hiddenbias[i] += biaslearnratehidden * biastrainamnt;
            }
        }
        #endregion

        #region StochasticLMTrain
        public void CalculateDiagonalHessian(INNInputProvider PProvider, int PHessianBatchLength)
        {
            double[][] input = new double[PHessianBatchLength][];
            double[][] desired = new double[PHessianBatchLength][];
            for (int i = 0; i < PHessianBatchLength; i++)
            {
                input[i] = PProvider.NextInputCase();
                desired[i] = PProvider.DesiredOutput();
            }
            double[][][] output = GetOutputFromInputsBatch(input);
            diagonalhessian = GetDerivativesFromOutputBatch(TDerivative.Second, output, input, desired);
            PProvider.Rewind(PHessianBatchLength);
        }
        public override void Train(INNInputProvider PProvider)
        {
            double[] input = PProvider.NextInputCase();
            double[][] output = GetOutputFromInputs(input);
            double[][][] derivatives = GetDerivativesFromOutput(TDerivative.First, output, input, PProvider.DesiredOutput());
            for (int i = 0; i < numlayers; i++)
            {
                for (int j = 0; j < layers[i].NumNeurons; j++)
                {
                    int numweightsback = weights[i][j].GetLength(0);
                    for (int k = 0; k < numweightsback; k++)
                    {
                        derivatives[i][j][k] *= (-learnrates[i]) / (diagonalhessian[i][j][k] + blowuppreventer);
                    }
                    derivatives[i][j][numweightsback] *= (-learnrates[i]) / (diagonalhessian[i][j][numweightsback] + blowuppreventer);
                }
            }
            ModifyWeights(derivatives);
        }
        private void ModifyWeights(double[][][] PAmount)
        {
            for (int i = 0; i < numlayers; i++)
            {
                double[] biases = layers[i].Biases;
                for (int j = 0; j < layers[i].NumNeurons; j++)
                {
                    int numweightsback = weights[i][j].GetLength(0);
                    for (int k = 0; k < numweightsback; k++)
                    {
                        weights[i][j][k] += PAmount[i][j][k];
                    }
                    biases[j] += PAmount[i][j][numweightsback];
                }
            }
        }
        #endregion

        #region Derivative
        private double[][][] GetDerivativesFromOutputBatch(TDerivative PWhichDeriv, double[][][] POutput,
                                                           double[][] PInput, double[][] PDesired)
        {
            double[][][] derivatives = ArrayInitOnePerWeight();
            Utility.ZeroArray(derivatives);
            int numcases = POutput.GetLength(0);
            for (int i = 0; i < numcases; i++)
            {
                Utility.AddArrays(derivatives, GetDerivativesFromOutput(PWhichDeriv, POutput[i], PInput[i], PDesired[i]));
            }
            Utility.ScaleArray(derivatives, 1 / ((double)numcases));
            return derivatives;
        }
        private double[][][] GetDerivativesFromOutput(TDerivative PWhichDeriv, double[][] POutput,
                                                      double[] PInput, double[] PDesired)
        {
            double[][][] derivatives = ArrayInitOnePerWeight();
            double[] layererror = LastLayerError(PWhichDeriv, POutput[numlayers - 1], PDesired);
            double[] previousoutput = PInput;
            if (numlayers > 1)
            {
                previousoutput = POutput[numlayers - 2];
            }
            CalculateDerivativesFromError(PWhichDeriv, numlayers - 1, derivatives[numlayers - 1], layererror, previousoutput);
            for(int i = numlayers - 2;i >= 0;i--)
            {
                layererror = GetLayerError(PWhichDeriv, i, POutput[i], layererror);
                previousoutput = PInput;
                if(i != 0)
                {
                    previousoutput = POutput[i - 1];
                }
                CalculateDerivativesFromError(PWhichDeriv, i, derivatives[i], layererror, previousoutput);
            }
            return derivatives;
        }
        private void CalculateDerivativesFromError(TDerivative PWhichDeriv, int PWhichLayer, 
                                                  double[][] PDerivatives, double[] PLayerError, double[] PPreviousOutputs)
        {
            int numneurons = layers[PWhichLayer].NumNeurons;
            int numneuronspreviouslayer = weights[PWhichLayer][0].GetLength(0);
            for (int i = 0; i < numneurons; i++)
            {
                for (int j = 0; j < numneuronspreviouslayer; j++)
                {
                    PDerivatives[i][j] = PLayerError[i] * PPreviousOutputs[j];
                    if (PWhichDeriv != TDerivative.First)
                    {
                        PDerivatives[i][j] *= PPreviousOutputs[j];
                    }
                }
                PDerivatives[i][numneuronspreviouslayer] = PLayerError[i];
            }
        }
        private double[] LastLayerError(TDerivative PWhichDeriv, double[] POutput,
                                        double[] PDesired)
        {
            int numneurons = layers[numlayers - 1].NumNeurons ;
            double[] error = new double[numneurons];
            for (int i = 0; i < numneurons; i++)
            {
                if (PWhichDeriv == TDerivative.First)
                {
                    error[i] = POutput[i] - PDesired[i];
                }
                else
                {
                    error[i] = 1;
                }
            }
            return error;
        }
        private double[] GetLayerError(TDerivative PWhichDeriv, int PWhichLayer,
                                       double[] POutput, double[] PNextLayerError)
        {
            int numneurons = layers[PWhichLayer].NumNeurons;
            int numneuronsnextlayer = layers[PWhichLayer + 1].NumNeurons;
            double[][] layerweights = weights[PWhichLayer + 1];
            double[] error = new double[numneurons];
            for (int i = 0; i < numneurons; i++)
            {
                error[i] = 0;
                for (int j = 0; j < numneuronsnextlayer; j++)
                {
                    double curweight = layerweights[j][i];
                    double addtoerror = PNextLayerError[j] * curweight;
                    if (PWhichDeriv != TDerivative.First)
                    {
                        addtoerror *= curweight;
                    }
                    error[i] += addtoerror;
                }
                double deriv = layers[PWhichLayer].FirstDerivative(POutput[i]);
                error[i] *= deriv;
                if (PWhichDeriv != TDerivative.First)
                {
                    error[i] *= deriv;
                }
            }
            return error;
        }
        private double[][][] ArrayInitOnePerWeight()
        {
            double[][][] derivatives = new double[numlayers][][];
            for (int i = 0; i < numlayers; i++)
            {
                int numneurons = layers[i].NumNeurons;
                derivatives[i] = new double[numneurons][];
                for (int j = 0; j < numneurons; j++)
                {
                    int numweightsperneuron = weights[i][j].GetLength(0);
                    //IMPORTANT: numweightsperneuron + ****1**** ----> accounts for the bias
                    derivatives[i][j] = new double[numweightsperneuron + 1];
                }
            }
            return derivatives;
        }
        #endregion

        #region Process
        public override void Run(INNInputProvider PProvider)
        {
            ProcessNetwork(PProvider.NextInputCase());
        }
        public void Run(double[] PData)
        {
            ProcessNetwork(PData);
        }
        private void SetInput(double[] PInput)
        {
            double[] output = inputlayer.Outputs;
            for (int i = 0; i < numinputs; i++)
            {
                output[i] = PInput[i];
            }
        }
        private void ProcessNetwork(double[] PInput)
        {
            double[] input = (double[])PInput.Clone();
            SetInput(PInput);
            for (int i = 0; i < numlayers; i++)
            {
                ProcessLayerForwards(i, input);
                input = layers[i].Outputs;
            }
        }
        private void ProcessLayerForwards(int PWhichLayer, double[] PLayerInputs)
        {
            int numunitsinlayer = layers[PWhichLayer].NumNeurons;
            int numinputstolayer = PLayerInputs.GetLength(0);
            for (int i = 0; i < numunitsinlayer; i++)
            {
                double unitinput = 0;
                for (int j = 0; j < numinputstolayer; j++)
                {
                    unitinput += weights[PWhichLayer][i][j] * PLayerInputs[j];
                }
                layers[PWhichLayer].Process(i, unitinput);
            }
        }
        private void ProcessLayerBackwards(int PWhichLayer, double[] PLayerInputs)
        {
            ILayer currentlayer = GetLayer(PWhichLayer);
            int numunitsinlayer = currentlayer.NumNeurons;
            int numinputstolayer = PLayerInputs.GetLength(0);
            for (int i = 0; i < numunitsinlayer; i++)
            {
                double unitinput = 0;
                for (int j = 0; j < numinputstolayer; j++)
                {
                    unitinput += weights[PWhichLayer + 1][j][i] * PLayerInputs[j];
                }
                currentlayer.Process(i, unitinput);
            }
        }
        private ILayer GetLayer(int PWhichLayer)
        {
            ILayer retval = inputlayer;
            if (PWhichLayer != -1)
            {
                retval = layers[PWhichLayer];
            }
            return retval;
        }
        public double[] GetLayerOutput(int PWhichLayer, double[] PInput)
        {
            double[] input = (double[])PInput.Clone();
            SetInput(PInput);
            for (int i = 0; i < PWhichLayer + 1; i++)
            {
                ProcessLayerForwards(i, input);
                input = layers[i].Outputs;
            }
            return (double[])input.Clone();
        }
        #endregion

        #region Output
        public double[][][] GetOutputFromInputsBatch(double[][] PInput)
        {
            int numcases = PInput.GetLength(0);
            double[][][] retval = new double[numcases][][];
            for (int i = 0; i < numcases; i++)
            {
                retval[i] = GetOutputFromInputs(PInput[i]);
            }
            return retval;
        }
        public double[][] GetOutputFromInputs(double[] PInput)
        {
            ProcessNetwork(PInput);
            double[][] retval = new double[numlayers][];
            for (int i = 0; i < numlayers; i++)
            {
                retval[i] = GetLayerOutput(i);
            }
            return retval;
        }
        public double[] GetLayerOutput(int PWhichLayer)
        {
            return (double[])layers[PWhichLayer].Outputs.Clone();;
        }
        public override double[] GetOutput()
        {
            return GetLayerOutput(numlayers - 1);
        }
        #endregion

        #region Save
        public void Save(string PFilename)
        {
            BinaryWriter file = new BinaryWriter(File.Create(PFilename));
            file.Write(numlayers);
            file.Write(numinputs);
            inputlayer.Write(file);
            SaveWeights(file);
            SaveLearnrates(file);
            SaveLayers(file);
            file.Close();
        }

        private void SaveWeights(BinaryWriter PFile)
        {
            int numweightsets = weights.GetLength(0);
            PFile.Write(numweightsets);
            for (int i = 0; i < numweightsets; i++)
            {
                int numfirstdimension = weights[i].GetLength(0);
                PFile.Write(numfirstdimension);
                for (int j = 0; j < numfirstdimension; j++)
                {
                    int numseconddimension = weights[i][j].GetLength(0);
                    PFile.Write(numseconddimension);
                    for (int k = 0; k < numseconddimension; k++)
                    {
                        PFile.Write(weights[i][j][k]);
                    }
                }
            }
        }

        private void SaveLearnrates(BinaryWriter PFile)
        {
            int numlearnrates = learnrates.GetLength(0);
            PFile.Write(numlearnrates);
            for (int i = 0; i < numlearnrates; i++)
            {
                PFile.Write(learnrates[i]);
            }
            PFile.Write(blowuppreventer);
        }

        private void SaveLayers(BinaryWriter PFile)
        {
            PFile.Write(numlayers);
            for (int i = 0; i < numlayers; i++)
            {
                layers[i].Write(PFile);
            }
        }
        #endregion

        #region Load
        public static Network Load(string PFilename)
        {
            BinaryReader file = new BinaryReader(File.OpenRead(PFilename));
            Network network = new Network();
            network.numlayers = file.ReadInt32();
            network.numinputs = file.ReadInt32();
            network.inputlayer = ILayer.Load(file);
            network.weights = LoadWeights(file);
            network.learnrates = LoadLearnrates(file);
            network.blowuppreventer = file.ReadDouble();
            network.layers = LoadLayers(file);
            return network;
        }

        private static ILayer[] LoadLayers(BinaryReader PFile)
        {
            int numlayers = PFile.ReadInt32();
            ILayer[] retval = new ILayer[numlayers];
            for (int i = 0; i < numlayers; i++)
            {
                retval[i] = ILayer.Load(PFile);
            }
            return retval;
        }

        private static double[] LoadLearnrates(BinaryReader PFile)
        {
            int numlearnrates = PFile.ReadInt32();
            double[] retval = new double[numlearnrates];
            for (int i = 0; i < numlearnrates; i++)
            {
                retval[i] = PFile.ReadDouble();
            }
            return retval;
        }

        private static double[][][] LoadWeights(BinaryReader PFile)
        {
            int numweightsets = PFile.ReadInt32();
            double[][][] retval = new double[numweightsets][][];
            for (int i = 0; i < numweightsets; i++)
            {
                int numfirstdimension = PFile.ReadInt32();
                retval[i] = new double[numfirstdimension][];
                for (int j = 0; j < numfirstdimension; j++)
                {
                    int numseconddimension = PFile.ReadInt32();
                    retval[i][j] = new double[numseconddimension];
                    for (int k = 0; k < numseconddimension; k++)
                    {
                        retval[i][j][k] = PFile.ReadDouble();
                    }
                }
            }
            return retval;
        }
        #endregion
    }
}

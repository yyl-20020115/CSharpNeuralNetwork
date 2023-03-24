
namespace Autoencoders
{
    /// <summary>   A training data. </summary>
    public struct TrainingData
    {
        /// <summary>   The position visible. </summary>
        public double[] posVisible;
        /// <summary>   The position hidden. </summary>
        public double[] posHidden;
        /// <summary>   The negative visible. </summary>
        public double[] negVisible;
        /// <summary>   The negative hidden. </summary>
        public double[] negHidden;

        /// <summary>   Zeroes this object. </summary>
        public void Zero()
        {
            Utility.SetArrayToZero(posVisible);
            Utility.SetArrayToZero(posHidden);
            Utility.SetArrayToZero(negVisible);
            Utility.SetArrayToZero(negHidden);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Scalars. </summary>
        ///
        /// <param name="PScalar">  The scalar. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void Scalar(double PScalar)
        {
            Utility.ScaleArray(posVisible, PScalar);
            Utility.ScaleArray(posHidden, PScalar);
            Utility.ScaleArray(negVisible, PScalar);
            Utility.ScaleArray(negHidden, PScalar);
        }
    }
    /// <summary>   A restricted boltzmann machine trainer. </summary>
    public static class RestrictedBoltzmannMachineTrainer
    {
        /// <summary>   The learnrate. </summary>
        private static RestrictedBoltzmannMachineLearningRate learnrate;
        /// <summary>   The weightset. </summary>
        private static RestrictedBoltzmannMachineWeightSet weightset;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Trains. </summary>
        ///
        /// <param name="PLayerVis">    The layer vis. </param>
        /// <param name="PLayerHid">    The layer HID. </param>
        /// <param name="PData">        The data. </param>
        /// <param name="PLearnRate">   The learn rate. </param>
        /// <param name="PWeightSet">   Set the weight belongs to. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static void Train(RestrictedBoltzmannMachineLayer PLayerVis, 
            RestrictedBoltzmannMachineLayer PLayerHid, TrainingData PData, 
            RestrictedBoltzmannMachineLearningRate PLearnRate, 
            RestrictedBoltzmannMachineWeightSet PWeightSet)
        {
            weightset = PWeightSet;
            learnrate = PLearnRate;
            for (int i = 0; i < PLayerVis.Count; i++)
            {
                for (int j = 0; j < PLayerHid.Count; j++)
                {
                    TrainWeight(i, j, CalculateTrainAmount(PData.posVisible[i], PData.posHidden[j], 
                        PData.negVisible[i], PData.negHidden[j]));
                }
                TrainBias(PLayerVis, i, PData.posVisible[i], PData.negVisible[i]);
            }
            for (int j = 0; j < PLayerHid.Count; j++)
            {
                TrainBias(PLayerHid, j, PData.posHidden[j], PData.negHidden[j]);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Calculates the train amount. </summary>
        ///
        /// <param name="PPosVis">  The position vis. </param>
        /// <param name="PPosHid">  The position HID. </param>
        /// <param name="PNegVis">  The negative vis. </param>
        /// <param name="PNegHid">  The negative HID. </param>
        ///
        /// <returns>   The calculated train amount. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private static double CalculateTrainAmount(double PPosVis, double PPosHid, double PNegVis, double PNegHid)
        {
            return ((PPosVis * PPosHid) - (PNegVis * PNegHid));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Train weight. </summary>
        ///
        /// <param name="PWhichVis">    The which vis. </param>
        /// <param name="PWhichHid">    The which HID. </param>
        /// <param name="PTrainAmount"> The train amount. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private static void TrainWeight(int PWhichVis, int PWhichHid, double PTrainAmount)
        {
            weightset.ModifyWeight(PWhichVis, PWhichHid, 
                 (learnrate.momentumWeights * weightset.GetWeightChange(PWhichVis, PWhichHid))
                  + (learnrate.weights * PTrainAmount) - (0.0002 * weightset.GetWeight(PWhichVis, PWhichHid)));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Train bias. </summary>
        ///
        /// <param name="PLayer">       The layer. </param>
        /// <param name="PWhich">       The which. </param>
        /// <param name="PPosPhase">    The position phase. </param>
        /// <param name="PNegPhase">    The negative phase. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private static void TrainBias(RestrictedBoltzmannMachineLayer PLayer, int PWhich, double PPosPhase, double PNegPhase)
        {
            double biaschange = (learnrate.momentumBiases * PLayer.GetBiasChange(PWhich)) + 
                                (learnrate.biases * (PPosPhase - PNegPhase));
            PLayer.SetBiasChange(PWhich, biaschange);
            PLayer.SetBias(PWhich, PLayer.GetBias(PWhich) + biaschange);
        }
    }
}

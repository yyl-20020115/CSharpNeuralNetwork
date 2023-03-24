using System.Collections.Generic;
using System.IO;

namespace Autoencoders
{
    /// <summary>   An autoencoder learning rate. </summary>
    public class AutoencoderLearningRate
    {
        #region Vars
        /// <summary>   The pre learning rate biases. </summary>
        internal List<double> preLearningRateBiases = new List<double>();
        /// <summary>   The pre learning rate weights. </summary>
        internal List<double> preLearningRateWeights = new List<double>();
        /// <summary>   The pre momentum biases. </summary>
        internal List<double> preMomentumBiases = new List<double>();
        /// <summary>   The pre momentum weights. </summary>
        internal List<double> preMomentumWeights = new List<double>();
        /// <summary>   The fine learning rate biases. </summary>
        internal List<double> fineLearningRateBiases = new List<double>();
        /// <summary>   The fine learning rate weights. </summary>
        internal List<double> fineLearningRateWeights = new List<double>();
        #endregion

        #region Accessors

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets pre training bias lr. </summary>
        ///
        /// <param name="PWhich">   The which. </param>
        ///
        /// <returns>   The pre training bias lr. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double GetPreTrainingBiasLR(int PWhich)
        {
            Utility.WithinBounds("Index out of bounds(Autoencoder)", PWhich, preLearningRateBiases.Count);
            return preLearningRateBiases[PWhich];
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets pre training weight lr. </summary>
        ///
        /// <param name="PWhich">   The which. </param>
        ///
        /// <returns>   The pre training weight lr. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double GetPreTrainingWeightLR(int PWhich)
        {
            Utility.WithinBounds("Index out of bounds(Autoencoder)", PWhich, preLearningRateWeights.Count);
            return preLearningRateWeights[PWhich];
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets pre training bias mom. </summary>
        ///
        /// <param name="PWhich">   The which. </param>
        ///
        /// <returns>   The pre training bias mom. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double GetPreTrainingBiasMom(int PWhich)
        {
            Utility.WithinBounds("Index out of bounds(Autoencoder)", PWhich, preMomentumBiases.Count);
            return preMomentumBiases[PWhich];
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets pre training weight mom. </summary>
        ///
        /// <param name="PWhich">   The which. </param>
        ///
        /// <returns>   The pre training weight mom. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double GetPreTrainingWeightMom(int PWhich)
        {
            Utility.WithinBounds("Index out of bounds(Autoencoder)", PWhich, preMomentumWeights.Count);
            return preMomentumWeights[PWhich];
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets fine tuning bias lr. </summary>
        ///
        /// <param name="PWhich">   The which. </param>
        ///
        /// <returns>   The fine tuning bias lr. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double GetFineTuningBiasLR(int PWhich)
        {
            Utility.WithinBounds("Index out of bounds(Autoencoder)", PWhich, fineLearningRateBiases.Count);
            return fineLearningRateBiases[PWhich];
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets fine tuning weight lr. </summary>
        ///
        /// <param name="PWhich">   The which. </param>
        ///
        /// <returns>   The fine tuning weight lr. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double GetFineTuningWeightLR(int PWhich)
        {
            Utility.WithinBounds("Index out of bounds(Autoencoder)", PWhich, fineLearningRateWeights.Count);
            return fineLearningRateWeights[PWhich];
        }
        #endregion

        #region Save/Load

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Saves the given p file. </summary>
        ///
        /// <param name="PFile">    The file to load. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal void Save(TextWriter PFile)
        {
            Utility.SaveArray(preLearningRateBiases?.ToArray() , PFile);
            Utility.SaveArray(preLearningRateWeights?.ToArray() , PFile);
            Utility.SaveArray(preMomentumBiases?.ToArray() , PFile);
            Utility.SaveArray(preMomentumWeights?.ToArray() , PFile);
            Utility.SaveArray(fineLearningRateBiases?.ToArray() , PFile);
            Utility.SaveArray(fineLearningRateWeights?.ToArray() , PFile);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Loads the given p file. </summary>
        ///
        /// <param name="PFile">    The file to load. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal void Load(TextReader PFile)
        {
            preLearningRateBiases?.AddRange(Utility.LoadArray(PFile));
            preLearningRateWeights?.AddRange(Utility.LoadArray(PFile));
            preMomentumBiases?.AddRange(Utility.LoadArray(PFile));
            preMomentumWeights?.AddRange(Utility.LoadArray(PFile));
            fineLearningRateBiases?.AddRange(Utility.LoadArray(PFile));
            fineLearningRateWeights?.AddRange(Utility.LoadArray(PFile));
        }
        #endregion
    }
}

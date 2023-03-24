
namespace Autoencoders
{
    /// <summary>   A restricted boltzmann machine learning rate. </summary>
    public struct RestrictedBoltzmannMachineLearningRate
    {
        /// <summary>   The weights. </summary>
        internal double weights;
        /// <summary>   The biases. </summary>
        internal double biases;
        /// <summary>   The momentum weights. </summary>
        internal double momentumWeights;
        /// <summary>   The momentum biases. </summary>
        internal double momentumBiases;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the Autoencoders.RestrictedBoltzmannMachineLearningRate struct.
        /// </summary>
        ///
        /// <param name="PLRWeights">   The plr weights. </param>
        /// <param name="PLRBiases">    The plr biases. </param>
        /// <param name="PMomWeights">  The mom weights. </param>
        /// <param name="PMomBiases">   The mom biases. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public RestrictedBoltzmannMachineLearningRate(double PLRWeights, double PLRBiases, double PMomWeights, double PMomBiases)
        {
            weights = PLRWeights;
            biases = PLRBiases;
            momentumWeights = PMomWeights;
            momentumBiases = PMomBiases;
        }
    }
}

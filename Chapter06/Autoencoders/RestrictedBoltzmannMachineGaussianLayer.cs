
namespace Autoencoders
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   A restricted boltzmann machine gaussian layer. </summary>
    ///
    /// <seealso cref="T:Autoencoders.RestrictedBoltzmannMachineLayer"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class RestrictedBoltzmannMachineGaussianLayer:RestrictedBoltzmannMachineLayer
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the Autoencoders.RestrictedBoltzmannMachineGaussianLayer
        /// class.
        /// </summary>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public RestrictedBoltzmannMachineGaussianLayer()
        {

        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the Autoencoders.RestrictedBoltzmannMachineGaussianLayer
        /// class.
        /// </summary>
        ///
        /// <param name="PSize">    The size. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public RestrictedBoltzmannMachineGaussianLayer(int PSize)
            : base(PSize)
        {

        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets a state. </summary>
        ///
        /// <param name="PWhich">   The which. </param>
        /// <param name="PInput">   The input. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override void SetState(int PWhich, double PInput)
        {
            CheckBounds(PWhich);
            activity[PWhich] = PInput + bias[PWhich];
            state[PWhich] = activity[PWhich] + Utility.NextGaussian(0, 1);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Creates a new object that is a copy of the current instance. </summary>
        ///
        /// <returns>   A new object that is a copy of this instance. </returns>
        ///
        /// <seealso cref="M:System.ICloneable.Clone()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override object Clone()
        {
            RestrictedBoltzmannMachineGaussianLayer retval = new RestrictedBoltzmannMachineGaussianLayer(numNeurons)
            {
                state = (double[])state.Clone(),
                bias = (double[])bias.Clone(),
                biasChange = (double[])biasChange.Clone(),
                activity = (double[])activity.Clone()
            };
            return retval;
        }
    }
}

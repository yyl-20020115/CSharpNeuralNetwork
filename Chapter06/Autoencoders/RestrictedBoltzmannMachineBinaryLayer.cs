using System;

namespace Autoencoders;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   A restricted boltzmann machine binary layer. </summary>
///
/// <seealso cref="T:Autoencoders.RestrictedBoltzmannMachineLayer"/>
////////////////////////////////////////////////////////////////////////////////////////////////////

public class RestrictedBoltzmannMachineBinaryLayer: RestrictedBoltzmannMachineLayer
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// Initializes a new instance of the Autoencoders.RestrictedBoltzmannMachineBinaryLayer
    /// class.
    /// </summary>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public RestrictedBoltzmannMachineBinaryLayer()
    {

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// Initializes a new instance of the Autoencoders.RestrictedBoltzmannMachineBinaryLayer
    /// class.
    /// </summary>
    ///
    /// <param name="PSize">    The size. </param>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public RestrictedBoltzmannMachineBinaryLayer(int PSize)
        :base(PSize)
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
        activity[PWhich] = 1 / (1 + Math.Exp(-PInput + bias[PWhich]));
        if (Utility.NextDouble() < activity[PWhich])
        {
            state[PWhich] = 1;
        }
        else
        {
            state[PWhich] = 0;
        }
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
        RestrictedBoltzmannMachineBinaryLayer retval = new RestrictedBoltzmannMachineBinaryLayer(numNeurons)
        {
            state = (double[])state.Clone(),
            bias = (double[])bias.Clone(),
            biasChange = (double[])biasChange.Clone(),
            activity = (double[])activity.Clone()
        };
        return retval;
    }
}

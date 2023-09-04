using System;
using System.IO;


////////////////////////////////////////////////////////////////////////////////////////////////////
// namespace: Autoencoders
//
// summary:	.
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace Autoencoders;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   An autoencoder weights. </summary>
///
/// <seealso cref="T:System.ICloneable"/>
////////////////////////////////////////////////////////////////////////////////////////////////////

public class AutoencoderWeights: ICloneable
{
    /// <summary>   The numweightsets. </summary>
    private int numweightsets;
    /// <summary>   The weights. </summary>
    private RestrictedBoltzmannMachineWeightSet[] weights;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// Initializes a new instance of the Autoencoders.AutoencoderWeights class.
    /// </summary>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public AutoencoderWeights()
    {

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// Initializes a new instance of the Autoencoders.AutoencoderWeights class.
    /// </summary>
    ///
    /// <param name="PNumLayers">       Number of layers. </param>
    /// <param name="PLayers">          The layers. </param>
    /// <param name="PWInitializer">    The password initializer. </param>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public AutoencoderWeights(int PNumLayers, RestrictedBoltzmannMachineLayer[] PLayers, IWeightInitializer PWInitializer)
    {
        numweightsets = PNumLayers - 1;
        weights = new RestrictedBoltzmannMachineWeightSet[numweightsets];
        for (int i = 0; i < numweightsets; i++)
        {
            weights[i] = new RestrictedBoltzmannMachineWeightSet(PLayers[i].Count, PLayers[i + 1].Count, PWInitializer);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Gets the number of weight sets. </summary>
    ///
    /// <value> The total number of weight sets. </value>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public int NumWeightSets => numweightsets;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Gets weight set. </summary>
    ///
    /// <param name="PPreSynapticLayer">    The pre synaptic layer. </param>
    ///
    /// <returns>   The weight set. </returns>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public RestrictedBoltzmannMachineWeightSet GetWeightSet(int PPreSynapticLayer)
    {
        Utility.WithinBounds("Invalid weight set index!", PPreSynapticLayer, numweightsets);
        return weights[PPreSynapticLayer];
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Gets recognition weight set. </summary>
    ///
    /// <param name="PPreSynapticLayer">    The pre synaptic layer. </param>
    ///
    /// <returns>   The recognition weight set. </returns>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public RestrictedBoltzmannMachineWeightSet GetRecogntionWeightSet(int PPreSynapticLayer)
    {
        Utility.WithinBounds("Invalid weight set index!", PPreSynapticLayer, numweightsets / 2);
        return weights[PPreSynapticLayer];
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Gets generative weight set. </summary>
    ///
    /// <param name="PPreSynapticLayer">    The pre synaptic layer. </param>
    ///
    /// <returns>   The generative weight set. </returns>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public RestrictedBoltzmannMachineWeightSet GetGenerativeWeightSet(int PPreSynapticLayer)
    {
        Utility.WithinBounds("Invalid weight set index!", PPreSynapticLayer, numweightsets / 2, numweightsets);
        return weights[PPreSynapticLayer];
    }

    #region Save/Load

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Saves the given p file. </summary>
    ///
    /// <param name="PFile">    The file to load. </param>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public void Save(TextWriter PFile)
    {
        PFile?.WriteLine(numweightsets);
        for (int i = 0; i < weights.GetLength(0); i++)
        {
            weights[i]?.Save(PFile);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Loads the given p file. </summary>
    ///
    /// <param name="PFile">    The file to load. </param>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public void Load(TextReader PFile)
    {
        numweightsets = int.Parse(PFile?.ReadLine());
        weights = new RestrictedBoltzmannMachineWeightSet[numweightsets];
        for(int i = 0;i < numweightsets;i++)
        {
            weights[i] = RestrictedBoltzmannMachineWeightSet.Load(PFile);
        }
    }
    #endregion

    #region ICloneable Members

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Creates a new object that is a copy of the current instance. </summary>
    ///
    /// <returns>   A new object that is a copy of this instance. </returns>
    ///
    /// <seealso cref="M:System.ICloneable.Clone()"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public object Clone()
    {
        var retval = new AutoencoderWeights
        {
            numweightsets = numweightsets,
            weights = new RestrictedBoltzmannMachineWeightSet[numweightsets]
        };
        for (int i = 0; i < numweightsets; i++)
        {
            retval.weights[i] = (RestrictedBoltzmannMachineWeightSet)weights[i].Clone();
        }
        return retval;
    }

    #endregion
}

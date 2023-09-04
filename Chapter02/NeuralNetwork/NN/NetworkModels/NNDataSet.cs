﻿namespace NeuralNetwork.NetworkModels;

/// <summary>   A nn data set. </summary>
public class NNDataSet
{

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Gets or sets the values. </summary>
    ///
    /// <value> The values. </value>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public readonly double[] Values;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Gets or sets the targets. </summary>
    ///
    /// <value> The targets. </value>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public readonly double[] Targets;


    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// Initializes a new instance of the NeuralNetwork.NetworkModels.NNDataSet class.
    /// </summary>
    ///
    /// <param name="values">   The values. </param>
    /// <param name="targets">  The targets. </param>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public NNDataSet(double[] values, double[] targets)
    {
        Values = values;
        Targets = targets;
    }
}

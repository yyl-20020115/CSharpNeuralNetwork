﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.NetworkModels;


/// <summary>   A neuron. </summary>
public class Neuron
{
    #region -- Properties --

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Gets or sets the identifier. </summary>
    ///
    /// <value> The identifier. </value>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public Guid Id { get; set; }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Gets or sets the input synapses. </summary>
    ///
    /// <value> The input synapses. </value>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public List<Synapse> InputSynapses { get; set; }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Gets or sets the output synapses. </summary>
    ///
    /// <value> The output synapses. </value>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public List<Synapse> OutputSynapses { get; set; }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Gets or sets the bias. </summary>
    ///
    /// <value> The bias. </value>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public double Bias { get; set; }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Gets or sets the bias delta. </summary>
    ///
    /// <value> The bias delta. </value>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public double BiasDelta { get; set; }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Gets or sets the gradient. </summary>
    ///
    /// <value> The gradient. </value>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public double Gradient { get; set; }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Gets or sets the value. </summary>
    ///
    /// <value> The value. </value>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public double Value { get; set; }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Gets or sets a value indicating whether this object is mirror. </summary>
    ///
    /// <value> True if this object is mirror, false if not. </value>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public bool IsMirror { get; set; }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Gets or sets a value indicating whether this object is canonical. </summary>
    ///
    /// <value> True if this object is canonical, false if not. </value>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public bool IsCanonical { get; set; }
    #endregion

    #region -- Constructors --

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// Initializes a new instance of the NeuralNetwork.NetworkModels.Neuron class.
    /// </summary>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public Neuron()
    {
        Id = Guid.NewGuid();
        InputSynapses = new ();
        OutputSynapses = new ();
        Bias = Network.GetRandom();
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// Initializes a new instance of the NeuralNetwork.NetworkModels.Neuron class.
    /// </summary>
    ///
    /// <param name="inputNeurons"> The input neurons. </param>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public Neuron(IEnumerable<Neuron> inputNeurons) : this()
    {
        //Ensure.That(inputNeurons).IsNotNull();

        foreach (var inputNeuron in inputNeurons)
        {
            var synapse = new Synapse(inputNeuron, this);
            inputNeuron?.OutputSynapses?.Add(synapse);
            InputSynapses?.Add(synapse);
        }
    }
    #endregion

    #region -- Values & Weights --

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Calculates the value. </summary>
    ///
    /// <returns>   The calculated value. </returns>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public virtual double CalculateValue()
    {
        return Value = Sigmoid.Output(InputSynapses.Sum(a => a.Weight * a.InputNeuron.Value) + Bias);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Calculates the error. </summary>
    ///
    /// <param name="target">   Target for the. </param>
    ///
    /// <returns>   The calculated error. </returns>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public double CalculateError(double target)
    {
        return target - Value;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Calculates the gradient. </summary>
    ///
    /// <param name="target">   (Optional) Target for the. </param>
    ///
    /// <returns>   The calculated gradient. </returns>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public double CalculateGradient(double? target = null)
    {
        if (target == null)
            return Gradient = OutputSynapses.Sum(a => a.OutputNeuron.Gradient * a.Weight) * Sigmoid.Derivative(Value);

        return Gradient = CalculateError(target.Value) * Sigmoid.Derivative(Value);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Updates the weights. </summary>
    ///
    /// <param name="learnRate">    The learn rate. </param>
    /// <param name="momentum">     The momentum. </param>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public void UpdateWeights(double learnRate, double momentum)
    {
        var prevDelta = BiasDelta;
        BiasDelta = learnRate * Gradient;
        Bias += BiasDelta + momentum * prevDelta;

        foreach (var synapse in InputSynapses)
        {
            prevDelta = synapse.WeightDelta;
            synapse.WeightDelta = learnRate * Gradient * synapse.InputNeuron.Value;
            synapse.Weight += synapse.WeightDelta + momentum * prevDelta;
        }
    }
    #endregion
}
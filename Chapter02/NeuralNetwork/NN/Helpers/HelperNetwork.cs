using System;
using System.Collections.Generic;

namespace NeuralNetwork.Helpers;

public class HelperNetwork
{
	public double LearningRate;
	public double Momentum;
	public List<NeuronData> InputLayer = new();
	public List<List<NeuronData>> HiddenLayers = new();
	public List<NeuronData> OutputLayer = new();
	public List<SynapseData> Synapses = new();
}

public class NeuronData
{
	public Guid Id;
	public double Bias;
	public double BiasDelta;
	public double Gradient;
	public double Value;
}

public class SynapseData
{
	public Guid Id;
	public Guid OutputNeuronId;
	public Guid InputNeuronId;
	public double Weight;
	public double WeightDelta;
}

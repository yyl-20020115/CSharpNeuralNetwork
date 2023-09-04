using System.IO;
using System.Windows.Forms;
using System.Collections.Generic;
using NeuralNetwork.NetworkModels;
using Newtonsoft.Json;

namespace NeuralNetwork.Helpers;

public static class ExportHelper
{
    public static void ExportNetwork(Network network)
    {
        var helperNetwork = GetHelperNetwork(network);

        using var dialog = new SaveFileDialog
        {
            Title = "Save Network File",
            Filter = "Text File|*.txt;"
        };

        if (dialog.ShowDialog() == DialogResult.OK)
        {
            using var file = File.CreateText(dialog.FileName);
            var serializer = new JsonSerializer { Formatting = Formatting.Indented };
            serializer.Serialize(file, helperNetwork);
        }
    }

    public static void ExportDatasets(List<NNDataSet> datasets)
    {
        using var dialog = new SaveFileDialog
        {
            Title = "Save Dataset File",
            Filter = "Text File|*.txt;"
        };

        if (dialog.ShowDialog() == DialogResult.OK)
        {
            using var file = File.CreateText(dialog.FileName);
            var serializer = new JsonSerializer { Formatting = Formatting.Indented };
            serializer.Serialize(file, datasets);
        }
    }

    private static HelperNetwork GetHelperNetwork(Network network)
    {
        //Ensure.That(network).IsNotNull();
        //Ensure.That(network.InputLayer).IsNotNull();
        //Ensure.That(network.HiddenLayers).IsNotNull();

        var helperNetwork = new HelperNetwork
        {
            LearningRate = network.LearningRate,
            Momentum = network.Momentum
        };

        //Input Layer
        foreach (var n in network.InputLayer)
        {
            var neuron = new NeuronData
            {
                Id = n.Id,
                Bias = n.Bias,
                BiasDelta = n.BiasDelta,
                Gradient = n.Gradient,
                Value = n.Value
            };

            helperNetwork.InputLayer?.Add(neuron);

            foreach (var synapse in n.OutputSynapses)
            {
                var syn = new SynapseData
                {
                    Id = synapse.Id,
                    OutputNeuronId = synapse.OutputNeuron.Id,
                    InputNeuronId = synapse.InputNeuron.Id,
                    Weight = synapse.Weight,
                    WeightDelta = synapse.WeightDelta
                };

                helperNetwork.Synapses?.Add(syn);
            }
        }

        //Hidden Layer
        foreach (var _layer in network.HiddenLayers)
        {
            var layer = new List<NeuronData>();

            foreach (var _neuron in _layer)
            {
                var neuron = new NeuronData
                {
                    Id = _neuron.Id,
                    Bias = _neuron.Bias,
                    BiasDelta = _neuron.BiasDelta,
                    Gradient = _neuron.Gradient,
                    Value = _neuron.Value
                };

                layer.Add(neuron);

                foreach (var synapse in _neuron.OutputSynapses)
                {
                    var syn = new SynapseData
                    {
                        Id = synapse.Id,
                        OutputNeuronId = synapse.OutputNeuron.Id,
                        InputNeuronId = synapse.InputNeuron.Id,
                        Weight = synapse.Weight,
                        WeightDelta = synapse.WeightDelta
                    };

                    helperNetwork.Synapses?.Add(syn);
                }
            }

            helperNetwork.HiddenLayers?.Add(layer);
        }

        //Output Layer
        foreach (var _neuron in network.OutputLayer)
        {
            var neuron = new NeuronData
            {
                Id = _neuron.Id,
                Bias = _neuron.Bias,
                BiasDelta = _neuron.BiasDelta,
                Gradient = _neuron.Gradient,
                Value = _neuron.Value
            };

            helperNetwork.OutputLayer?.Add(neuron);

            foreach (var synapse in _neuron.OutputSynapses)
            {
                var syn = new SynapseData
                {
                    Id = synapse.Id,
                    OutputNeuronId = synapse.OutputNeuron.Id,
                    InputNeuronId = synapse.InputNeuron.Id,
                    Weight = synapse.Weight,
                    WeightDelta = synapse.WeightDelta
                };

                helperNetwork.Synapses?.Add(syn);
            }
        }

        return helperNetwork;
    }
}
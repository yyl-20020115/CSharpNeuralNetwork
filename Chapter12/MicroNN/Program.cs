using System;

namespace MicroNN;

public class Program
{
    public static void Main(string[] args)
    {
        var n1 = new Neuron(1);
        var n2 = new Neuron(1);
        var n3 = new Neuron(1);
        n1.Fires += n3.Signal;
        n2.Fires += n3.Signal;
        n1.Reset();
        n2.Reset();
        n3.Reset();

    }
}

public sealed class Neuron
{
    public event Action<double> Fires = delegate { };

    public readonly double _threshold;
    private double _signalReceived;

    public Neuron(double threshold)
    {
        _threshold = threshold;
        Reset();
    }

    public void Reset() => _signalReceived = 0;
    public void Signal(double strength)
    {
        if ((_signalReceived += strength) >= _threshold)
            Fires(.5);
    }
}

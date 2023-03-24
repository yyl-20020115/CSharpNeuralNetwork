using System;

namespace NeuralNetwork;

public class Program
{
    static void Main(string[] args)
    {
        var mgr = new NNManager();
        mgr.SetupNetwork()
            .GetTrainingDataFromUser()
            .TrainNetworkToMinimum()
            .TestNetwork();

        Console.WriteLine("Press any key to train network for maximum");
        Console.ReadKey();

        mgr.SetupNetwork()
            .GetTrainingDataFromUser()
            .TrainNetworkToMaximum()
            .TestNetwork();

        Console.WriteLine("Press any key to exit");
        Console.ReadKey();
    }
}

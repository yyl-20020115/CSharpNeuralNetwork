using System;

namespace NeuralNetwork;

public class Program
{
    public static void Main(string[] _)
    {
        var manager = new NNManager();
        manager.SetupNetwork()
            .GetTrainingDataFromUser()
            .TrainNetworkToMinimum()
            .TestNetwork();

        Console.WriteLine("Press any key to train network for maximum");
        Console.ReadKey();

        manager.SetupNetwork()
            .GetTrainingDataFromUser()
            .TrainNetworkToMaximum()
            .TestNetwork();

        Console.WriteLine("Press any key to exit");
        Console.ReadKey();
    }
}

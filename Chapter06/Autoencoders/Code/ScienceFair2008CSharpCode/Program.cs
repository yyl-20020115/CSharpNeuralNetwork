using System;
using System.Collections.Generic;
using System.Text;
using ScienceFair2008.MNIST;

namespace ScienceFair2008
{
    class Program
    {
        static void Main(string[] args)
        {
            //OverComplete();

            //GenerativeACRun(1, 10);
            //GenerativeACRun(5, 10);
            //GenerativeACRun(10, 10);
            //GenerativeACRun(15, 10);
            //GenerativeACRun(30, 10);

            //LabeledDataTestACMLPBoth(10, "Net10epochsT10Out.Net", 50, 3000, 75, 75, 50, false);


            //LabeledDataTestACMLP(2000, "OverComplete1epochsT.Net", 50, 3000, 5, 5, 50, false);

            //MNISTAutoencoderCompression.Run("Net10epochsT5Out.Net", "5outputs");
            //MNISTAutoencoderCompression.Run("Net10epochsT10Out.Net", "10outputs");
            //MNISTAutoencoderCompression.Run("Net10epochsT15Out.Net", "15outputs");
            //MNISTAutoencoderCompression.Run("Net10epochsT30Out.Net", "30outputs");
            //MNISTAutoencoderCompression.SaveOriginalMNIST("original");
            
            //LabeledDataTestACMLP(5, "Net10epochsT5Out.Net", 50, 3000, 75, 75, 50, 1, false);
            //LabeledDataTestACMLP(5, "Net10epochsT5Out.Net", 100, 1500, 50, 50, 100, 1, false);
            //LabeledDataTestACMLP(5, "Net10epochsT5Out.Net", 250, 2000, 60, 60, 250, 1, false);
            //LabeledDataTestACMLP(5, "Net10epochsT5Out.Net", 500, 600, 20, 20, 150, 1, false);
            //LabeledDataTestACMLP(5, "Net10epochsT5Out.Net", 1000, 300, 10, 10, 250, 1, false);
            //LabeledDataTestACMLP(5, "Net10epochsT5Out.Net", 5000, 60, 5, 5, 1000, 1, false);
            //LabeledDataTestACMLP(5, "Net10epochsT5Out.Net", 20000, 15, 1, 1, 1000, 2, false);
            //LabeledDataTestACMLP(5, "Net10epochsT5Out.Net", 60000, 10, 1, 1, 1000, 6, false);

            //LabeledDataTestACMLP(10, "Net10epochsT10Out.Net", 50, 3000, 75, 75, 50, 1, false);
            //LabeledDataTestACMLP(10, "Net10epochsT10Out.Net", 100, 1500, 50, 50, 100, 1, false);
            //LabeledDataTestACMLP(10, "Net10epochsT10Out.Net", 250, 2000, 60, 60, 250, 1, false);
            //LabeledDataTestACMLP(10, "Net10epochsT10Out.Net", 500, 600, 20, 20, 150, 1, false);
            //LabeledDataTestACMLP(10, "Net10epochsT10Out.Net", 1000, 300, 10, 10, 250, 1, false);
            //LabeledDataTestACMLP(10, "Net10epochsT10Out.Net", 5000, 60, 5, 5, 1000, 1, false);
            //LabeledDataTestACMLP(10, "Net10epochsT10Out.Net", 20000, 15, 1, 1, 1000, 2, false);
            //LabeledDataTestACMLP(10, "Net10epochsT10Out.Net", 60000, 10, 1, 1, 1000, 6, false);

            //LabeledDataTestACMLP(15, "Net10epochsT15Out.Net", 50, 3000, 75, 75, 50, 1, false);
            //LabeledDataTestACMLP(15, "Net10epochsT15Out.Net", 100, 1500, 50, 50, 100, 1, false);
            //LabeledDataTestACMLP(15, "Net10epochsT15Out.Net", 250, 2000, 60, 60, 250, 1, false);
            //LabeledDataTestACMLP(15, "Net10epochsT15Out.Net", 500, 600, 20, 20, 150, 1, false);
            //LabeledDataTestACMLP(15, "Net10epochsT15Out.Net", 1000, 300, 10, 10, 250, 1, false);
            //LabeledDataTestACMLP(15, "Net10epochsT15Out.Net", 5000, 60, 5, 5, 1000, 1, false);
            //LabeledDataTestACMLP(15, "Net10epochsT15Out.Net", 20000, 15, 1, 1, 1000, 2, false);
            //LabeledDataTestACMLP(15, "Net10epochsT15Out.Net", 60000, 10, 1, 1, 1000, 6, false);

            //LabeledDataTestACMLP(30, "Net10epochsT30Out.Net", 50, 3000, 75, 75, 50, 1, false);
            //LabeledDataTestACMLP(30, "Net10epochsT30Out.Net", 100, 1500, 50, 50, 100, 1, false);
            //LabeledDataTestACMLP(30, "Net10epochsT30Out.Net", 250, 2000, 60, 60, 250, 1, false);
            //LabeledDataTestACMLP(30, "Net10epochsT30Out.Net", 500, 600, 20, 20, 150, 1, false);
            //LabeledDataTestACMLP(30, "Net10epochsT30Out.Net", 1000, 300, 10, 10, 250, 1, false);
            //LabeledDataTestACMLP(30, "Net10epochsT30Out.Net", 5000, 60, 5, 5, 1000, 1, false);
            //LabeledDataTestACMLP(30, "Net10epochsT30Out.Net", 20000, 15, 1, 1, 1000, 2, false);
            //LabeledDataTestACMLP(30, "Net10epochsT30Out.Net", 60000, 10, 1, 1, 1000, 6, false);

            //LabeledDataTestJustMLP(50, 3000, 75, 75, 50,false);
            //LabeledDataTestJustMLP(100, 500, 15, 15, 100,false);
            //LabeledDataTestJustMLP(250, 350, 15, 15, 250, false);
            //LabeledDataTestJustMLP(500, 600, 20, 20, 150,false);
            //LabeledDataTestJustMLP(1000, 300, 10, 10, 250,false);
            //LabeledDataTestJustMLP(5000, 60, 5, 5, 1000, false);
            //LabeledDataTestJustMLP(20000, 15, 1, 1, 1000, false);
            //LabeledDataTestJustMLP(60000, 5, 1, 1, 60000, false);
            
            for (int i = 0; i < 5; i++)
            {
                LabeledDataTestACMLP(5, "Net10epochsT5Out.Net", 50, 500, 75, 75, 50, 1, false,i.ToString());
            }
            

            //FlickrAC();
        }

        private static void OverComplete()
        {
            GenerativeAutoencoder program = new GenerativeAutoencoder();
            program.LoadNetName = null;// "Netstart.Net";
            int[] layers = new int[3];
            layers[0] = 500;
            layers[1] = 500;
            layers[2] = 2000;
            program.SavePrefix = "OverComplete";
            program.LayerSizes = layers;
            program.IntervalTesting = true;
            program.NumTrainingEpochs = 1;
            program.Init();
            program.Run();
            program.Display();
        }

        private static void FlickrAC()
        {
            FlickrAutoencoder program = new FlickrAutoencoder();
            string[] tags = new string[1];
            tags[0] = "food";
            //tags[1] = "road";
            //tags[2] = "cloud";
            //tags[3] = "scene";
            //tags[4] = "mountain";
            program.HessianSize = 500;
            program.LoadNetName = "FlickrAC500epochsT.Net";
            program.SavePrefix = "FlickrAC";
            program.NumTestingCases = 500;
            program.SaveInterval = 1;
            program.NumTrainingEpochs = 1000;
            program.NumTrainingCases = 500;
            program.NumPreTrainEpochs = 1;
            program.Tags = tags;
            program.IntervalTesting = false;
            program.Init();
            //program.Run();
            program.Display();
        }

        private static void GenerativeACRun(int PNumOutputs, int PNumEpochs)
        {
            GenerativeAutoencoder program = new GenerativeAutoencoder();
            program.LoadNetName = null;// "Netstart.Net";
            int[] layers = new int[4];
            layers[0] = 500;
            layers[1] = 300;
            layers[2] = 125;
            layers[3] = PNumOutputs;
            program.LayerSizes = layers;
            program.IntervalTesting = true;
            program.NumTrainingEpochs = PNumEpochs;
            program.Init();
            program.Run();
            program.Display();
        }

        private static void LabeledDataTestACMLPBoth(int PNumAutoencoderOutputs, string PAutoencoderFile,
                                            int PNumLabeled, int PNumEpochs,
                                            int PSaveInterval, int PTestInterval,
                                            int PHessianSize, bool PIdealized)
        {
            AutoencoderBasedDiscrimination program = new AutoencoderBasedDiscrimination();
            program.IntervalTesting = true;
            program.SaveInterval = PSaveInterval;
            program.BaseNetName = PAutoencoderFile;
            program.LoadNetName = null;
            program.SavePrefix = PNumLabeled + "Lbled" + PNumAutoencoderOutputs + "OutsACB";
            program.NumTrainingEpochs = PNumEpochs;
            program.HessianCalculationsPerEpoch = 1;
            program.HessianSize = PHessianSize;
            program.NumTrainingCases = PNumLabeled;
            program.TestIntervalLength = PTestInterval;
            if (PIdealized)
            {
                program.TrainingSet = "idealizedtrainingset.bmp";
            }
            program.Init();
            program.Run();
        }

        private static void LabeledDataTestACMLP(int PNumAutoencoderOutputs, string PAutoencoderFile,
                                            int PNumLabeled, int PNumEpochs,
                                            int PSaveInterval, int PTestInterval,
                                            int PHessianSize, int PNumHessianCalcs, bool PIdealized, string PName)
        {
            AutoencoderToMLP program = new AutoencoderToMLP();
            program.IntervalTesting = true;
            program.SaveInterval = PSaveInterval;
            program.AutoencoderFilename = PAutoencoderFile;
            program.AutoencoderOutputs = PNumAutoencoderOutputs;
            program.LoadNetName = null;
            program.SavePrefix = PNumLabeled + "Lbled" + PNumAutoencoderOutputs + "Outs" + PName;
            program.NumTrainingEpochs = PNumEpochs;
            program.HessianCalculationsPerEpoch = PNumHessianCalcs;
            program.HessianSize = PHessianSize;
            program.NumTrainingCases = PNumLabeled;
            program.TestIntervalLength = PTestInterval;
            if (PIdealized)
            {
                program.TrainingSet = "idealizedtrainingset.bmp";
            }
            program.Init();
            program.Run();
        }

        private static void LabeledDataTestJustMLP(int PNumLabeled, int PNumEpochs,
                                            int PSaveInterval, int PTestInterval,
                                            int PHessianSize, bool PIdealized)
        {
            DiscriminativeMLP program2 = new DiscriminativeMLP();
            program2.IntervalTesting = true;
            program2.LoadNetName = null; 
            program2.SaveInterval = PSaveInterval;
            program2.SavePrefix = "Net" + PNumLabeled + "TrainingLabeled";
            program2.NumTrainingEpochs = PNumEpochs;
            program2.HessianCalculationsPerEpoch = 1;
            program2.HessianSize = PHessianSize;
            program2.NumTrainingCases = PNumLabeled;
            program2.TestIntervalLength = PTestInterval;
            if (PIdealized)
            {
                program2.TrainingSet = "idealizedtrainingset.bmp";
            }
            program2.Init();
            program2.Run();
        }
    }
}

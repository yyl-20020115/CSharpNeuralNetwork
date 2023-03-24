using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork;
using System.Drawing;
using System.IO;

namespace ScienceFair2008.MNIST
{
    public static class LabelReader
    {
        public static int[] ReadLabels(string PFileName)
        {
            if (!File.Exists(PFileName))
            {
                throw new Exception("Blaaaaaah....");
            }
            int[] retval;
            BinaryReader file = new BinaryReader(File.Open(PFileName, FileMode.Open));
            file.ReadByte();
            file.ReadByte();
            file.ReadByte();
            file.ReadByte();
            file.ReadByte();
            file.ReadByte();
            file.ReadByte();
            file.ReadByte();
            int numitems = 10000;
            if (PFileName == "traininglabels")
            {
                numitems = 60000;
            }
            retval = new int[numitems];
            for (int i = 0; i < numitems; i++)
            {
                retval[i] = file.ReadByte();
            }
            file.Close();
            return retval;
        }
    }
    public static class MNISTAutoencoderCompression
    {
        static Network ac;
        static Bitmap trainingset;
        static Bitmap testingset;
        static double[][] trainingdata;
        static double[][] testingdata;
        static int[] traininglabels;
        static int[] testinglabels;
        static int imagesize = 28;
        static int numtrainimages = 60000;
        static int numtestimages = 10000;

        static double[][] outputtrain;
        static double[][] outputtest;
        public static void Run(string PAutoencoderFile, string POutputFilePrefix)
        {
            LoadAutoencoder(PAutoencoderFile);
            LoadLabels();
            InitTrainingData();
            SaveNewDataSet();

            StartSaveData(POutputFilePrefix);
        }

        public static void SaveOriginalMNIST(string POutputFilePrefix)
        {
            LoadLabels();
            InitTrainingData();
            outputtrain = new double[numtrainimages][];
            for (int i = 0; i < numtrainimages; i++)
            {
                outputtrain[i] = trainingdata[i];
                Console.WriteLine(i + "Train");
            }
            outputtest = new double[numtestimages][];
            for (int i = 0; i < numtestimages; i++)
            {
                outputtest[i] = testingdata[i];
                Console.WriteLine(i + "Test");
            }

            StartSaveData(POutputFilePrefix);
        }

        private static void StartSaveData(string POutputFilePrefix)
        {
            TextWriter file = new StreamWriter("training" + POutputFilePrefix + ".data");
            SaveData(false, file);
            file.Close();


            file = new StreamWriter("testing" + POutputFilePrefix + ".data");
            SaveData(true, file);
            file.Close();
        }
        private static void LoadLabels()
        {
            traininglabels = LabelReader.ReadLabels("traininglabels");
            testinglabels = LabelReader.ReadLabels("testinglabels");
        }
        private static void SaveNewDataSet()
        {
            outputtrain = new double[numtrainimages][];
            for (int i = 0; i < numtrainimages; i++)
            {
                outputtrain[i] = ac.GetLayerOutput(ac.NumLayers / 2 - 1, trainingdata[i]);
                Console.WriteLine(i + "Train");
            }
            outputtest = new double[numtestimages][];
            for (int i = 0; i < numtestimages; i++)
            {
                outputtest[i] = ac.GetLayerOutput(ac.NumLayers / 2 - 1, testingdata[i]);
                Console.WriteLine(i + "Test");
            }
        }

        private static void SaveData(bool PTesting, TextWriter PFile)
        {
            double[][] output = (PTesting) ? outputtest : outputtrain;
            /*int numoutputs = ac.Layers[ac.NumLayers / 2 - 1].NumNeurons;
            int[] maxindex = new int[numoutputs];
            double[] max = new double[numoutputs];
            int[] minindex = new int[numoutputs];
            double[] min = new double[numoutputs];
            for (int i = 0; i < output.GetLength(0); i++)
            {
                for (int j = 0; j < numoutputs; j++)
                {
                    if (output[i][j] > output[maxindex[j]][j])
                    {
                        maxindex[j] = i;
                    }
                    if (output[i][j] < output[minindex[j]][j])
                    {
                        minindex[j] = i;
                    }
                }
            }
            for (int i = 0; i < numoutputs; i++)
            {
                max[i] = output[maxindex[i]][i];
                min[i] = output[minindex[i]][i];
            }
            for (int i = 0; i < output.GetLength(0); i++)
            {
                for (int j = 0; j < numoutputs; j++)
                {
                    output[i][j] = (output[i][j] - min[j]) / (max[j] - min[j]);
                }
            }*/
            string fileout1 = "0";
            for (int j = 1; j < output[0].GetLength(0); j++)
            {
                fileout1 = fileout1 + "," + j.ToString();
            }
            fileout1 = fileout1 + "," + "Labels";
            PFile.WriteLine(fileout1);
            for (int i = 0; i < output.GetLength(0); i++)
            {
                string fileout = output[i][0].ToString();
                for (int j = 1; j < output[i].GetLength(0); j++)
                {
                    fileout = fileout + "," + output[i][j].ToString();
                }
                if (PTesting)
                {
                    fileout = fileout + "," + testinglabels[i];
                }
                else
                {
                    fileout = fileout + "," + traininglabels[i];
                }
                PFile.WriteLine(fileout);
            }
        }

        private static void LoadAutoencoder(string PLoadFilename)
        {
            ac = Network.Load(PLoadFilename);
        }
        private static void InitTrainingData()
        {
            trainingset = new Bitmap("trainingset" + imagesize + ".bmp");
            testingset = new Bitmap("testingset" + imagesize + ".bmp");
            trainingdata = new double[numtrainimages][];
            for (int j = 0; j < numtrainimages; j++)
            {
                trainingdata[j] = PictureData(j, trainingset);
            }
            testingdata = new double[numtestimages][];
            for (int j = 0; j < numtestimages; j++)
            {
                testingdata[j] = PictureData(j, testingset);
            }
        }
        static double[] PictureData(int PWhich, Bitmap PSet)
        {
            int xposition = PWhich % (PSet.Width / imagesize);
            int yposition = (PWhich - xposition) / (PSet.Width / imagesize);
            int xinitialpos = xposition * imagesize;
            int yinitialpos = yposition * imagesize;
            double[] data = new double[imagesize * imagesize];
            for (int i = 0; i < imagesize; i++)
            {
                for (int j = 0; j < imagesize; j++)
                {
                    int xpos = i + xinitialpos;
                    int ypos = j + yinitialpos;
                    Color color = PSet.GetPixel(xpos, ypos);
                    data[i * imagesize + j] = (((double)1) / ((double)255)) * (255 - ((double)color.R));
                    /*if (color.R != 255)
                    {
                        data[i * imagesize + j] = 1;
                    }
                    else
                    {
                        data[i * imagesize + j] = 0;
                    }*/
                }
            }
            return data;
        }
    }
}

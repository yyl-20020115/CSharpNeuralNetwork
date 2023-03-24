using System;
using System.Collections.Generic;
using System.Text;
using NNBase;
using System.Drawing;

namespace ScienceFair2008
{
    public abstract class MNISTInputProvider:INNInputProvider
    {
        protected Bitmap trainingset;
        protected Bitmap testingset;
        protected int[] traininglabels;
        protected int[] testinglabels;
        protected int imagesize;
        protected int trainingsetsize;
        protected int testingsetsize;
        protected int trainingindex = 0;
        protected int testingindex = 0;
        protected bool testing = false;
        protected double[] curdata;

        public int TestingIndex
        {
            get { return testingindex; }
        }
        public int TrainingIndex
        {
            get { return trainingindex; }
        }
        public bool Testing
        {
            get { return testing; }
            set { testing = value; }
        }
        public int Imagesize
        {
            get { return imagesize; }
        }
        public int TestingSetSize
        {
            get { return testingsetsize; }
        }
        public int TrainingSetSize
        {
            get { return trainingsetsize; }
        }
        public int CurrentSetSize
        {
            get
            {
                if (testing)
                {
                    return testingsetsize;
                }
                return trainingsetsize;
            }
        }


        public MNISTInputProvider(string PTrainingFileName, string PTestingFileName,
                                  string PTrainingLabelFile, string PTestingLabelFile,
                                  int PTrainingSetSize, int PTestingSetSize,
                                  int PImageSize)
        {
            LoadImageSets(PTrainingFileName, PTestingFileName);
            LoadLabels(PTrainingLabelFile, PTestingLabelFile);
            trainingsetsize = PTrainingSetSize;
            testingsetsize = PTestingSetSize;
            imagesize = PImageSize;
            curdata = PictureData(0, trainingset, imagesize);
        }
        protected virtual void LoadImageSets(string PTrainingFileName, string PTestingFileName)
        {
            trainingset = new Bitmap(PTrainingFileName);
            testingset = new Bitmap(PTestingFileName);
        }
        protected virtual void LoadLabels(string PTrainingLabelFile, string PTestingLabelFile)
        {
            traininglabels = LabelReader.ReadLabels(PTrainingLabelFile, true);
            testinglabels = LabelReader.ReadLabels(PTestingLabelFile, false);
        }

        #region INNInputProvider Members
        public virtual double[] NextInputCase()
        {
            if (testing)
            {
                if (testingindex >= testingsetsize)
                {
                    testingindex = 0;
                }
                testingindex++;
                curdata = PictureData(testingindex - 1, testingset, imagesize);
            }
            else
            {
                if (trainingindex >= trainingsetsize)
                {
                    trainingindex = 0;
                }
                trainingindex++;
                curdata = PictureData(trainingindex - 1, trainingset, imagesize);
            }
            return curdata;
        }
        public abstract double[] DesiredOutput();
        public virtual void Rewind(int PAmount)
        {
            ChangeIndexPosition(-PAmount);
        }
        public virtual void FastForward(int PAmount)
        {
            ChangeIndexPosition(PAmount);
        }
        public virtual void ChangeIndexPosition(int PAmount)
        {
            if (testing)
            {
                testingindex += PAmount;
                if (testingindex < 0)
                {
                    testingindex = 0;
                }
                else if (testingindex >= testingsetsize)
                {
                    trainingindex = 0;
                }
            }
            else
            {
                trainingindex += PAmount;
                if (trainingindex < 0)
                {
                    trainingindex = 0;
                }
                else if (trainingindex >= trainingsetsize)
                {
                    trainingindex = 0;
                }
            }
        }
        #endregion

        protected virtual double[] PictureData(int PWhich, Bitmap PSet, int PImageSize)
        {
            int xposition = PWhich % (PSet.Width / PImageSize);
            int yposition = (PWhich - xposition) / (PSet.Width / PImageSize);
            int xinitialpos = xposition * PImageSize;
            int yinitialpos = yposition * PImageSize;
            double[] data = new double[PImageSize * PImageSize];
            for (int i = 0; i < PImageSize; i++)
            {
                for (int j = 0; j < PImageSize; j++)
                {
                    int xpos = i + xinitialpos;
                    int ypos = j + yinitialpos;
                    Color color = PSet.GetPixel(xpos, ypos);
                    data[i * PImageSize + j] = (((double)1) / ((double)255)) * (255 - ((double)color.R));
                }
            }
            return data;
        }
    }
}

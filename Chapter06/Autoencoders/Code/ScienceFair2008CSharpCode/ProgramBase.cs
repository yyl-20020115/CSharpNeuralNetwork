using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork;
using Totem.CameraSystem;
using Totem.Input;
using Totem.Core;
using System.IO;
using System.Drawing;

namespace ScienceFair2008
{
    public abstract class ProgramBase
    {
        protected string loadnetname = null;
        protected int numtrainingcases = 60000;
        protected int numtestingcases = 10000;
        protected int numtrainingepochs = 1;
        protected int hessiansize = 500;
        
        protected string saveprefix = "Net";
        protected int saveinterval = 1;
        protected Network net;


        protected TotemCore Core;
        protected InputManager Input;
        protected CameraSystem camsys;
        protected int displayindex = 0;
        protected double[] curoutputs = null;

        public string LoadNetName
        {
            get { return loadnetname; }
            set { loadnetname = value; }
        }
        public int NumTrainingCases
        {
            get { return numtrainingcases; }
            set { numtrainingcases = value; }
        }
        public int NumTestingCases
        {
            get { return numtestingcases; }
            set { numtestingcases = value; }
        }
        public int NumTrainingEpochs
        {
            get { return numtrainingepochs; }
            set { numtrainingepochs = value; }
        }
        public int HessianSize
        {
            get { return hessiansize; }
            set { hessiansize = value; }
        }
        public string SavePrefix
        {
            get { return saveprefix; }
            set { saveprefix = value; }
        }
        public int SaveInterval
        {
            get { return saveinterval; }
            set { saveinterval = value; }
        }


        public abstract void Init();
        protected void SetNetLearnrates(double PLearnrate, double PBlowUpPreventer)
        {
            double[] learnrates = net.Learnrates;
            for (int i = 0; i < learnrates.GetLength(0); i++)
            {
                learnrates[i] = PLearnrate;
            }
            net.BlowUpPreventer = PBlowUpPreventer;
        }
        
        public abstract void Run();

        public abstract void Display();

        public abstract string CreateSpecificFilePath(string PFileName);
        public virtual string CreateDataFilePath(string PFileName)
        {
            return ("Data\\" + PFileName);
        }
    }
}

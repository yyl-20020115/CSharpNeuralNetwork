using System;
using System.Collections.Generic;
using System.Text;
using System.Drawing;
using Totem.Core;
using Totem.Input;
using Totem.CameraSystem;
using Tao.OpenGl;
using Sharp3D.Math.Core;
using NNBase;
using NeuralNetwork;
using System.IO;

namespace ScienceFair2008
{
    public abstract class MNISTProgramBase:ProgramBase
    {
        protected string trainingset = "trainingset28.bmp";
        protected string testingset = "testingset28.bmp";
        protected string traininglabels = "traininglabels";
        protected string testinglabels = "testinglabels";
        protected int hessiancalculationsperepoch = 6;
        protected int imagesize = 28;
        protected bool intervaltesting = true;
        protected int testintervallength = 1;

        protected MNISTInputProvider inputprovider;

        public string TrainingSet
        {
            get { return trainingset; }
            set { trainingset = value; }
        }
        public string TestingSet
        {
            get { return testingset; }
            set { testingset = value; }
        }
        public string TrainingLabels
        {
            get { return traininglabels; }
            set { traininglabels = value; }
        }
        public string TestingLabels
        {
            get { return testinglabels; }
            set { testinglabels = value; }
        }
        public int HessianCalculationsPerEpoch
        {
            get { return hessiancalculationsperepoch; }
            set { hessiancalculationsperepoch = value; }
        }
        public int ImageSize
        {
            get { return imagesize; }
            set { imagesize = value; }
        }
        public bool IntervalTesting
        {
            get { return intervaltesting; }
            set { intervaltesting = value; }
        }
        public int TestIntervalLength
        {
            get { return testintervallength; }
            set { testintervallength = value; }
        }

        public override void Run()
        {
            TryIntervalTest();
            for (int i = 0; i < numtrainingepochs; i++)
            {
                for (int j = 0; j < hessiancalculationsperepoch; j++)
                {
                    int numcases = (int)Math.Floor(((double)numtrainingcases) / ((double)hessiancalculationsperepoch));
                    if (j == hessiancalculationsperepoch - 1)
                    {
                        numcases += numtrainingcases % numcases;
                    }
                    double mse = Train(numcases);
                    WriteOneLineFile(saveprefix + "trainingerror.txt", mse);
                }
                if (i % saveinterval == 0)
                {
                    net.Save(CreateSpecificFilePath(saveprefix + (i + 1) + "epochsT.Net"));
                }
                if ((i % testintervallength) == 0 && (i != 0 || testintervallength == 1))
                {
                    TryIntervalTest();
                }
            }
        }

        protected void TryIntervalTest()
        {
            if (intervaltesting)
            {
                double mse = Test(10000);
                WriteOneLineFile(saveprefix + "testingerror.txt", mse);
            }
        }
        protected void WriteOneLineFile(string PFileName, double PLine)
        {
            TextWriter testfile = new StreamWriter(CreateSpecificFilePath(PFileName), true);
            testfile.WriteLine(PLine);
            testfile.Close();
        }

        protected virtual double Train(int PNumCases)
        {
            CalculateHessian();
            double mse = 0;
            for (int i = 0; i < PNumCases; i++)
            {
                net.Train(inputprovider);
                double[] desiredoutput = inputprovider.DesiredOutput();
                mse += net.CalculateMSE(net.NumLayers - 1, desiredoutput);
                Console.WriteLine(mse / (i + 1) + "     " + i);
            }
            return mse / PNumCases;
        }
        protected void CalculateHessian()
        {
            net.CalculateDiagonalHessian(inputprovider, hessiansize);
        }
        protected virtual double Test()
        {
            return Test(10000);
        }
        protected virtual double Test(int PNumCases)
        {
            inputprovider.Testing = true;
            double mse = 0;
            for (int i = 0; i < PNumCases; i++)
            {
                net.Run(inputprovider);
                double[] desiredoutput = inputprovider.DesiredOutput();
                mse += net.CalculateMSE(net.NumLayers - 1, desiredoutput);
                Console.WriteLine(mse / (i + 1) + "     " + i);
            }
            inputprovider.Testing = false;
            return mse / PNumCases;
        }

        public override void Display()
        {
            InitTotem();
            Core.Run();
        }

        #region Rendering
        protected virtual void Input_OnClick(Point pos, System.Windows.Forms.MouseButtons which)
        {
            displayindex++;
            if (displayindex == inputprovider.CurrentSetSize)
            {
                displayindex = 0;
            }
            SetCurrentOutputs();
        }
        protected virtual void Input_OnKeyDown(Microsoft.DirectX.DirectInput.Key which)
        {
            if (which == Microsoft.DirectX.DirectInput.Key.Escape)
            {
                Core.Kill();
            }
        }
        protected virtual void Core_OnRender()
        {
            int imagesize = inputprovider.Imagesize;
            double[] originaldata = SetCurrentInput();
            Gl.glBegin(Gl.GL_QUADS);
            for (int i = 0; i < imagesize; i++)
            {
                for (int j = 0; j < imagesize; j++)
                {
                    Gl.glColor3d(originaldata[i * imagesize + j], originaldata[i * imagesize + j], originaldata[i * imagesize + j]);
                    Gl.glVertex3f((float)i, (float)-j, 0);
                    Gl.glVertex3f((float)i + 1, (float)-j, 0);
                    Gl.glVertex3f((float)i + 1, (float)-j - 1, 0);
                    Gl.glVertex3f((float)i, (float)-j - 1, 0);
                    Gl.glColor3d(curoutputs[i * imagesize + j], curoutputs[i * imagesize + j], curoutputs[i * imagesize + j]);
                    Gl.glVertex3f((float)i + imagesize, (float)-j, 0);
                    Gl.glVertex3f((float)i + imagesize + 1, (float)-j, 0);
                    Gl.glVertex3f((float)i + imagesize + 1, (float)-j - 1, 0);
                    Gl.glVertex3f((float)i + imagesize, (float)-j - 1, 0);
                }
            }
            Gl.glEnd();
        }
        protected virtual void InitTotem()
        {
            int imagesize = inputprovider.Imagesize;
            GraphicsConfig configinfo = new GraphicsConfig();
            configinfo.Default();
            configinfo.Far = 1000.0F;
            Core = new TotemCore("Window", configinfo);
            Input = new InputManager(Core.OpenGl.OGLForm);
            Input.OnClick += new InputManager.OnClickHandler(Input_OnClick);
            Input.OnKeyDown += new InputManager.OnKeyDownHandler(Input_OnKeyDown);
            camsys = new CameraSystem(new MNISTCamera(new Vector3F(((float)imagesize * 2.0f) / 2.0f, ((float)-imagesize) / 2.0f, ((float)imagesize) * 3.0f)
                , new Vector3F(((float)imagesize * 2.0f) / 2.0f, -((float)imagesize) / 2.0f, 0)));
            Core.PluginSystem.AddPlugin(Input);
            Core.PluginSystem.AddPlugin(camsys);
            Core.OnRender += new TotemCore.DMRenderHandler(Core_OnRender);
            curoutputs = SetCurrentOutputs();
        }


        protected virtual double[] SetCurrentOutputs()
        {
            //done so that if display doesn't use totem then it won't break
            throw new Exception("This method must be implemented in a base class.");
        }

        protected virtual double[] SetCurrentInput()
        {
            //done so that if display doesn't use totem then it won't break
            throw new Exception("This method must be implemented in a base class.");
        }
        #endregion
    }
}

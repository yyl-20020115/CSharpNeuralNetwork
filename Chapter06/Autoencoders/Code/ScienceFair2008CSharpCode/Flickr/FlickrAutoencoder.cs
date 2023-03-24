using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork;
using System.Drawing;
using Tao.OpenGl;
using Totem.Core;
using Totem.Input;
using Totem.CameraSystem;
using Sharp3D.Math.Core;
using System.IO;

namespace ScienceFair2008
{
    public class FlickrAutoencoder: ProgramBase
    {
        private FlickrInputProvider inputprovider;
        private string[] tags;
        private int numpretrainepochs;
        private bool intervaltesting = true;
        private int testintervallength = 1;
        private int hessiancalculationsperepoch = 1;

        public bool IntervalTesting
        {
            get { return intervaltesting; }
            set { intervaltesting = value; }
        }
        public int NumPreTrainEpochs
        {
            get { return numpretrainepochs; }
            set { numpretrainepochs = value; }
        }
        public string[] Tags
        {
            get { return tags; }
            set { tags = value; }
        }

        public override void Init()
        {
            InitInputProvider();
            LoadNetwork();
            SetNetLearnrates(0.001, 0.1);
        }

        private void InitInputProvider()
        {
            inputprovider = new FlickrInputProvider(tags, 200);
        }
        private void LoadNetwork()
        {
            if (loadnetname != null)
            {
                net = Network.Load(CreateSpecificFilePath(loadnetname));
            }
            else
            {
                net = CreatePreTrainedNet();
            }
        }

        #region PreTraining
        private Network CreatePreTrainedNet()
        {
            Network pretrainnet = InitPreTrainedNet();
            PreTrain(pretrainnet, numtrainingcases);
            return PreTrainComplete(pretrainnet);
        }
        private Network InitPreTrainedNet()
        {
            ILayer[] layers = new ILayer[1];
            layers[0] = new SigmoidLayer(100, 0.001);
            //layers[1] = new SigmoidLayer(numoutputs, 0.05);
            //layers[2] = new SigmoidLayer(100, 0.01);
            //layers[3] = new SigmoidLayer(numoutputs, 0.01);
            double[] learnrates = new double[1];
            learnrates[0] = 0.001;
            //learnrates[1] = 0.05;
            //learnrates[2] = 0.01;
            //learnrates[3] = 0.01;
            return new Network(new SigmoidLayer(625, 0.001), layers, learnrates);
        }
        private void PreTrain(Network PPreTrainNet, int PNumCases)
        {
            for (int i = 0; i < PPreTrainNet.NumLayers; i++)
            {
                inputprovider.ResetPageAndIndex();
                for (int j = 0; j < PNumCases * numpretrainepochs; j++)
                {
                    PPreTrainNet.PreTrain(inputprovider, i);
                    Console.WriteLine("       " + j);
                }
            }
        }
        private Network PreTrainComplete(Network PPreTrainNet)
        {
            PPreTrainNet.Save(CreateSpecificFilePath("pretrain.net"));
            int numlayerspt = PPreTrainNet.NumLayers;
            int numlayersnew = numlayerspt * 2;
            ILayer inputlayer = PPreTrainNet.InputLayer.Clone();
            ILayer[] layers = new ILayer[numlayersnew];
            double[] learnrate = new double[numlayersnew];
            double[][][] weights = new double[numlayersnew][][];
            for (int i = 0; i < numlayerspt; i++)
            {
                ILayer copylayer = PPreTrainNet.Layers[i];
                layers[i] = copylayer.Clone();
                learnrate[i] = PPreTrainNet.Learnrates[i];
                weights[i] = (double[][])PPreTrainNet.Weights[i].Clone();
            }
            for (int i = 0; i < numlayerspt - 1; i++)
            {
                ILayer copylayer = PPreTrainNet.Layers[i];
                int copydestination = numlayerspt * 2 - i - 2;
                layers[copydestination] = copylayer.Clone();
                learnrate[copydestination] = PPreTrainNet.Learnrates[i];
            }
            for (int i = 0; i < numlayerspt; i++)
            {
                int copydestination = numlayerspt * 2 - i - 1;
                weights[copydestination] = Utility.TransposeArray(PPreTrainNet.Weights[i]);
            }
            layers[numlayersnew - 1] = PPreTrainNet.InputLayer.Clone();
            learnrate[numlayersnew - 1] = PPreTrainNet.Learnrates[0];
            weights[numlayersnew - 1] = Utility.TransposeArray(PPreTrainNet.Weights[0]);
            return new Network(inputlayer, layers, weights, null, learnrate);
        }
        #endregion


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
        private double Train(int PNumCases)
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
        private void TryIntervalTest()
        {
            if (intervaltesting)
            {
                double mse = Test(numtestingcases);
                WriteOneLineFile(saveprefix + "testingerror.txt", mse);
            }
        }
        private void WriteOneLineFile(string PFileName, double PLine)
        {
            TextWriter testfile = new StreamWriter(CreateSpecificFilePath(PFileName), true);
            testfile.WriteLine(PLine);
            testfile.Close();
        }
        private void CalculateHessian()
        {
            net.CalculateDiagonalHessian(inputprovider, hessiansize);
        }
        private double Test(int PNumCases)
        {
            double mse = 0;
            for (int i = 0; i < PNumCases; i++)
            {
                net.Run(inputprovider);
                double[] desiredoutput = inputprovider.DesiredOutput();
                mse += net.CalculateMSE(net.NumLayers - 1, desiredoutput);
                Console.WriteLine(mse / (i + 1) + "     " + i);
            }
            return mse / PNumCases;
        }

        public override void Display()
        {
            InitTotem();
            Core.Run();
        }

        #region Rendering
        private void Input_OnClick(Point pos, System.Windows.Forms.MouseButtons which)
        {
            displayindex++;
            curoutputs = SetCurrentOutputs();
        }
        private void Input_OnKeyDown(Microsoft.DirectX.DirectInput.Key which)
        {
            if (which == Microsoft.DirectX.DirectInput.Key.Escape)
            {
                Core.Kill();
            }
        }
        private void Core_OnRender()
        {
            int imagesizex = inputprovider.ImageSizeX;
            int imagesizey = inputprovider.ImageSizeY;
            double[] originaldata = SetCurrentInput();
            Gl.glBegin(Gl.GL_QUADS);
            for (int i = 0; i < imagesizex; i++)
            {
                for (int j = 0; j < imagesizey; j++)
                {
                    Gl.glColor3d(originaldata[i * imagesizey + j], originaldata[i * imagesizey + j], originaldata[i * imagesizey + j]);
                    Gl.glVertex3f((float)i, (float)-j, 0);
                    Gl.glVertex3f((float)i + 1, (float)-j, 0);
                    Gl.glVertex3f((float)i + 1, (float)-j - 1, 0);
                    Gl.glVertex3f((float)i, (float)-j - 1, 0);
                    Gl.glColor3d(curoutputs[i * imagesizey + j], curoutputs[i * imagesizey + j], curoutputs[i * imagesizey + j]);
                    Gl.glVertex3f((float)i + imagesizey + 10, (float)-j, 0);
                    Gl.glVertex3f((float)i + imagesizey + 11, (float)-j, 0);
                    Gl.glVertex3f((float)i + imagesizey + 11, (float)-j - 1, 0);
                    Gl.glVertex3f((float)i + imagesizey + 10, (float)-j - 1, 0);
                }
            }
            Gl.glEnd();
        }
        private void InitTotem()
        {
            int imagesizex = inputprovider.ImageSizeX;
            int imagesizey = inputprovider.ImageSizeY;
            GraphicsConfig configinfo = new GraphicsConfig();
            configinfo.Default();
            configinfo.Far = 1000.0F;
            Core = new TotemCore("Window", configinfo);
            Input = new InputManager(Core.OpenGl.OGLForm);
            Input.OnClick += new InputManager.OnClickHandler(Input_OnClick);
            Input.OnKeyDown += new InputManager.OnKeyDownHandler(Input_OnKeyDown);
            camsys = new CameraSystem(new MNISTCamera(new Vector3F(((float)imagesizex * 2.0f) / 2.0f, ((float)-imagesizey) / 2.0f,
                (float)(Math.Sqrt(((double)imagesizex) * ((double)imagesizey)) * 3.0)), 
                new Vector3F(((float)imagesizex * 2.0f) / 2.0f, -((float)imagesizey) / 2.0f, 0)));
            Core.PluginSystem.AddPlugin(Input);
            Core.PluginSystem.AddPlugin(camsys);
            Core.OnRender += new TotemCore.DMRenderHandler(Core_OnRender);
            curoutputs = SetCurrentOutputs();
        }

        private double[] SetCurrentOutputs()
        {
            net.Run(inputprovider);
            return net.GetOutput();
        }
        private double[] SetCurrentInput()
        {
            return inputprovider.DesiredOutput();
        }
        #endregion

        public override string CreateSpecificFilePath(string PFileName)
        {
            return ("Data\\FlickrAutoencoder\\" + PFileName);
        }
    }
}

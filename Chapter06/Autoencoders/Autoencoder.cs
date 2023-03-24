using System;
using System.Collections.Generic;
using System.IO;

namespace Autoencoders
{
    /// <summary>   An autoencoder. </summary>
    public class Autoencoder
    {
        #region Vars
        /// <summary>   The numlayers. </summary>
        private int numlayers;
        /// <summary>   True to pretraining. </summary>
        private bool pretraining = true;
        /// <summary>   The layers. </summary>
        private RestrictedBoltzmannMachineLayer[] layers;
        /// <summary>   The learnrate. </summary>
        private AutoencoderLearningRate learnrate;
        /// <summary>   The recognitionweights. </summary>
        private AutoencoderWeights recognitionweights;
        /// <summary>   The generativeweights. </summary>
        private AutoencoderWeights generativeweights;
        /// <summary>   The trainingdata. </summary>
        private TrainingData[] trainingdata;
        /// <summary>   The errorobservers. </summary>
        private List<IErrorObserver> errorobservers;
        #endregion

        #region Initialization

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Prevents a default instance of the Autoencoders.Autoencoder class from being created.
        /// </summary>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private Autoencoder()
        {
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes a new instance of the Autoencoders.Autoencoder class. </summary>
        ///
        /// <param name="PLayers">          The layers. </param>
        /// <param name="PTrainingInfo">    Information describing the training. </param>
        /// <param name="PWInitializer">    The password initializer. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal Autoencoder(List<RestrictedBoltzmannMachineLayer> PLayers, AutoencoderLearningRate PTrainingInfo
            , IWeightInitializer PWInitializer)
        {
            numlayers = PLayers.Count;
            layers = PLayers.ToArray();
            learnrate = PTrainingInfo;
            recognitionweights = new AutoencoderWeights(numlayers, layers, PWInitializer);
            generativeweights = new AutoencoderWeights(numlayers, layers, PWInitializer);
            errorobservers = new List<IErrorObserver>();
            InitializeBiases(PWInitializer);
            InitializeTrainingData();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes the biases. </summary>
        ///
        /// <param name="PWInitializer">    The password initializer. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void InitializeBiases(IWeightInitializer PWInitializer)
        {
            for (int i = 0; i < numlayers; i++)
            {
                for (int j = 0; j < layers[i].Count; j++)
                {
                    layers[i].SetBias(j, PWInitializer.InitializeBias());
                }
            }
        }
        /// <summary>   Initializes the training data. </summary>
        private void InitializeTrainingData()
        {
            trainingdata = new TrainingData[numlayers - 1];
            for (int i = 0; i < numlayers - 1; i++)
            {
                trainingdata[i].posVisible = new double[layers[i].Count];
                Utility.SetArrayToZero(trainingdata[i].posVisible);
                trainingdata[i].posHidden = new double[layers[i + 1].Count];
                Utility.SetArrayToZero(trainingdata[i].posHidden);
                trainingdata[i].negVisible = new double[layers[i].Count];
                Utility.SetArrayToZero(trainingdata[i].negVisible);
                trainingdata[i].negHidden = new double[layers[i + 1].Count];
                Utility.SetArrayToZero(trainingdata[i].negHidden);
            }
        }
        #endregion

        #region Accessors

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the number of layers. </summary>
        ///
        /// <value> The total number of layers. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public int NumLayers => numlayers;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets recognition weights. </summary>
        ///
        /// <returns>   The recognition weights. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public AutoencoderWeights GetRecognitionWeights()
        {
            return recognitionweights;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets generative weights. </summary>
        ///
        /// <returns>   The generative weights. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public AutoencoderWeights GetGenerativeWeights()
        {
            return generativeweights;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets a layer. </summary>
        ///
        /// <param name="PWhichLayer">  The which layer. </param>
        ///
        /// <returns>   The layer. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public RestrictedBoltzmannMachineLayer GetLayer(int PWhichLayer)
        {
            Utility.WithinBounds("Layer index out of bounds!", PWhichLayer, numlayers);
            return layers[PWhichLayer];
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the learning rate. </summary>
        ///
        /// <value> The learning rate. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public AutoencoderLearningRate LearningRate => learnrate;

        #endregion

        #region PreTraining

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Pre train. </summary>
        ///
        /// <param name="PWhichLayer">  The which layer. </param>
        /// <param name="PData">        The data. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void PreTrain(int PWhichLayer, double[] PData)
        {
            double[][] sentdata = new double[1][];
            sentdata[0] = PData;
            PreTrain(PWhichLayer, sentdata, 1);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Pre train. </summary>
        ///
        /// <exception cref="Exception">    Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="PWhichLayer">  The which layer. </param>
        /// <param name="PData">        The data. </param>
        /// <param name="PBatchSize">   Size of the batch. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void PreTrain(int PWhichLayer, double[][] PData, int PBatchSize)
        {
            if (!pretraining)
            {
                throw new AutoEncoderException("You can only call PreTrainingComplete() once!");
            }
            Utility.WithinBounds("Layer to pretrain is invalid!", PWhichLayer, numlayers);
            Utility.WithinBounds("Invalid pre training batch size!", PBatchSize, PData.GetLength(0) + 1);
            double[][][] batches = CreateBatches(CalculateToLayer(0,PWhichLayer, PData), PBatchSize);
          
            for (int i = 0; i < batches.GetLength(0); i++)
            {
                PreTrainCollectData(PWhichLayer, batches[i]);
                PerformPreTraining(PWhichLayer);
                CalculateError(trainingdata[PWhichLayer].posVisible, trainingdata[PWhichLayer].negVisible);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Pre training complete. </summary>
        ///
        /// <exception cref="Exception">    Thrown when an exception error condition occurs. </exception>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void PreTrainingComplete()
        {
            if (!pretraining)
            {
                throw new AutoEncoderException("You can only call PreTrainingComplete() once!");
            }
            pretraining = false;
            for (int i = 0; i < numlayers; i++)
            {
                generativeweights = (AutoencoderWeights)recognitionweights.Clone();
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Pre train collect data. </summary>
        ///
        /// <param name="PWhichLayer">  The which layer. </param>
        /// <param name="PData">        The data. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void PreTrainCollectData(int PWhichLayer, double[][] PData)
        {
            trainingdata[PWhichLayer].Zero();
            
            for(int i = 0;i < PData.GetLength(0);i++)
            {
                SetLayerData(PWhichLayer,PData[i]);
                UpdateLayer(PWhichLayer + 1, true, recognitionweights);
                Utility.AddArrays(trainingdata[PWhichLayer].posVisible, layers[PWhichLayer].GetActivities());
                Utility.AddArrays(trainingdata[PWhichLayer].posHidden, layers[PWhichLayer + 1].GetActivities());
                UpdateLayer(PWhichLayer, false, recognitionweights);
                UpdateLayer(PWhichLayer + 1, true, recognitionweights);
                Utility.AddArrays(trainingdata[PWhichLayer].negVisible, layers[PWhichLayer].GetActivities());
                Utility.AddArrays(trainingdata[PWhichLayer].negHidden, layers[PWhichLayer + 1].GetActivities());
            }
            trainingdata[PWhichLayer].Scalar(1 / PData.GetLength(0));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Performs the pre training action. </summary>
        ///
        /// <param name="PPreSynapticLayer">    The pre synaptic layer. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void PerformPreTraining(int PPreSynapticLayer)
        {
            RestrictedBoltzmannMachineLearningRate sentlearnrate = new RestrictedBoltzmannMachineLearningRate(
                learnrate.preLearningRateWeights[PPreSynapticLayer], 
                learnrate.preLearningRateBiases[PPreSynapticLayer], 
                learnrate.preMomentumWeights[PPreSynapticLayer], 
                learnrate.preMomentumBiases[PPreSynapticLayer]);
            RestrictedBoltzmannMachineTrainer.Train(layers[PPreSynapticLayer], layers[PPreSynapticLayer + 1], 
                trainingdata[PPreSynapticLayer], sentlearnrate, recognitionweights.GetWeightSet(PPreSynapticLayer));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Updates the layer backward pre train described by PWhich. </summary>
        ///
        /// <param name="PWhich">   The which. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void UpdateLayerBackwardPreTrain(int PWhich)
        {
            Utility.WithinBounds("Cannot update this layer!!!", PWhich, numlayers - 1);
            RestrictedBoltzmannMachineLayer thislayer = layers[PWhich];
            RestrictedBoltzmannMachineLayer nextlayer = layers[PWhich + 1];
            double input = 0;
            double[] states = nextlayer.GetStates();
            for (int i = 0; i < thislayer.Count; i++)
            {
                for (int j = 0; j < nextlayer.Count; j++)
                {
                    input += recognitionweights.GetWeightSet(PWhich).GetWeight(i, j) * states[j];
                }
                thislayer.SetState(i, input);
                input = 0;
            }
        }
        #endregion

        #region FineTuning
        /*
         * Implements Wake-Sleep Training for an autoencoder. 
         */

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Fine tune. </summary>
        ///
        /// <param name="PData">    The data. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void FineTune(double[] PData)
        {
            double[][] sentdata = new double[1][];
            sentdata[0] = PData;
            FineTune(sentdata, 1);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Fine tune. </summary>
        ///
        /// <exception cref="Exception">    Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="PData">        The data. </param>
        /// <param name="PBatchSize">   Size of the batch. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void FineTune(double[][] PData, int PBatchSize)
        {
            if (pretraining)
            {
                throw new AutoEncoderException("You cannot FineTune until calling PreTrainingComplete()!");
            }
            if (PBatchSize < 1 || PBatchSize > PData.GetLength(0))
            {
                throw new ArgumentException("Invalid pre training batch size!");
            }
            double[][][] batches = CreateBatches(PData, PBatchSize);
            for (int i = 0; i < batches.GetLength(0); i++)
            {
                FineTuneBatch(batches[i]);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Fine tune batch. </summary>
        ///
        /// <param name="PData">    The data. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void FineTuneBatch(double[][] PData)
        {
            for(int i = 0;i < PData.GetLength(0);i++)
            {
                Compress(PData[i]);
                WakePhase();
                Reconstruct();
                SleepPhase();
                CalculateError(PData[i], layers[0].GetActivities());
            }
        }

        /// <summary>   Wake phase. </summary>
        private void WakePhase()
        {
            for (int i = 0; i < numlayers - 1; i++)
            {
                double[] visstates = layers[i].GetStates();
                double[] visact = layers[i].GetActivities();
                double[] hidstates = layers[i + 1].GetStates();
                double[] hidact = layers[i + 1].GetActivities();
                double curlearnrate = learnrate.fineLearningRateWeights[i];
                for (int j = 0; j < layers[i].Count; j++)
                {
                    for (int k = 0; k < layers[i + 1].Count; k++)
                    {
                        generativeweights.GetWeightSet(i).ModifyWeight(j, k, curlearnrate *
                               (hidstates[k] * (visstates[j] - visact[j])));
                    }
                }
            }
        }
        /// <summary>   Sleep phase. </summary>
        private void SleepPhase()
        {
            for (int i = 0; i < numlayers - 1; i++)
            {
                double[] visstates = layers[i].GetStates();
                double[] visact = layers[i].GetActivities();
                double[] hidstates = layers[i + 1].GetStates();
                double[] hidact = layers[i + 1].GetActivities();
                double curlearnrate = learnrate.fineLearningRateWeights[i];
                for (int j = 0; j < layers[i].Count; j++)
                {
                    for (int k = 0; k < layers[i + 1].Count; k++)
                    {
                        recognitionweights.GetWeightSet(i).ModifyWeight(j, k, curlearnrate *
                               (visstates[j] * (hidstates[k] - hidact[k])));
                    }
                }
            }
        }


        #endregion

        #region Running

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Compress the given p data. </summary>
        ///
        /// <exception cref="Exception">    Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="PData">    The data. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void Compress(double[] PData)
        {
            if (PData == null)
            {
                throw new ArgumentNullException("No null data allowed!");
            }
            CalculateToLayer(0, numlayers - 1, PData);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Reconstructs this object. </summary>
        ///
        /// <exception cref="Exception">    Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="PData">    The data. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void Reconstruct(double[] PData)
        {
            if(PData == null)
            {
                throw new ArgumentNullException("No null data allowed!");
            }
            CalculateToLayer(numlayers - 1, 0, PData);
        }
        /// <summary>   Reconstructs this object. </summary>
        public void Reconstruct()
        {
            Reconstruct(layers[numlayers - 1].GetStates());
        }
        #endregion

        #region LayerCalculation

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Calculates to layer. </summary>
        ///
        /// <param name="PStartLayer">  The start layer. </param>
        /// <param name="PEndLayer">    The end layer. </param>
        /// <param name="PData">        The data. </param>
        ///
        /// <returns>   The calculated to layer. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private double[][] CalculateToLayer(int PStartLayer, int PEndLayer, double[][] PData)
        {
            int batchsize = PData.GetLength(0);
            double[][] newdata = new double[batchsize][];
            for (int i = 0; i < batchsize; i++)
            {
                newdata[i] = CalculateToLayer(PStartLayer, PEndLayer, PData[i]);
            }
            return newdata;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Calculates to layer. </summary>
        ///
        /// <param name="PStartLayer">  The start layer. </param>
        /// <param name="PEndLayer">    The end layer. </param>
        /// <param name="PData">        The data. </param>
        ///
        /// <returns>   The calculated to layer. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private double[] CalculateToLayer(int PStartLayer, int PEndLayer, double[] PData)
        {
            SetLayerData(PStartLayer, PData);
            return CalculateToLayer(PStartLayer, PEndLayer);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Calculates to layer. </summary>
        ///
        /// <param name="PStartLayer">  The start layer. </param>
        /// <param name="PEndLayer">    The end layer. </param>
        ///
        /// <returns>   The calculated to layer. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private double[] CalculateToLayer(int PStartLayer, int PEndLayer)
        {
            int looplimit = PEndLayer - PStartLayer;
            if (PStartLayer > PEndLayer)
            {
                looplimit = PStartLayer - PEndLayer;
            }
            for (int i = 1; i <= looplimit; i++)
            {
                if (PStartLayer < PEndLayer)
                {
                    UpdateLayer(PStartLayer + i, true, recognitionweights);
                }
                else
                {
                    UpdateLayer(PStartLayer - i, false, generativeweights);
                }
            }
            return layers[PEndLayer].GetActivities();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Updates the layer. </summary>
        ///
        /// <param name="PWhichLayer">  The which layer. </param>
        /// <param name="PForward">     True to forward. </param>
        /// <param name="PWeights">     The weights. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void UpdateLayer(int PWhichLayer, bool PForward, AutoencoderWeights PWeights)
        {
            int beginlayer = PWhichLayer - 1;
            if (PForward)
            {
                Utility.WithinBounds("Cannot update this layer!!!", PWhichLayer, 1, numlayers);
            }
            else
            {
                Utility.WithinBounds("Cannot update this layer!!!", PWhichLayer, 0, numlayers - 1);
                beginlayer = PWhichLayer + 1;
            }
            RestrictedBoltzmannMachineLayer thislayer = layers[PWhichLayer];
            RestrictedBoltzmannMachineLayer previouslayer = layers[beginlayer];
            double input = 0;
            double[] states = previouslayer.GetStates();
            for (int i = 0; i < thislayer.Count; i++)
            {
                for (int j = 0; j < previouslayer.Count; j++)
                {
                    if (!PForward)
                    {
                        input += PWeights.GetWeightSet(beginlayer - 1).GetWeight(i, j) * states[j];
                    }
                    else
                    {
                        input += PWeights.GetWeightSet(beginlayer).GetWeight(j, i) * states[j];
                    }
                }
                thislayer.SetState(i, input);
                input = 0;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets layer data. </summary>
        ///
        /// <exception cref="ArgumentException">    Thrown when one or more arguments have unsupported or
        ///                                         illegal values. </exception>
        ///
        /// <param name="PWhich">   The which. </param>
        /// <param name="PData">    The data. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void SetLayerData(int PWhich, double[] PData)
        {
            Utility.WithinBounds("Layer index out of bounds!", PWhich, numlayers);
            if (PData.GetLength(0) != layers[PWhich].Count)
            {
                throw new ArgumentException("Too little or too much initial data");
            }
            for (int i = 0; i < layers[PWhich].Count; i++)
            {
                layers[PWhich].SetStateBypass(i, PData[i]);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Creates the batches. </summary>
        ///
        /// <param name="PData">        The data. </param>
        /// <param name="PBatchSize">   Size of the batch. </param>
        ///
        /// <returns>   A new array of double[][]. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private double[][][] CreateBatches(double[][] PData, int PBatchSize)
        {
            int numbatches = (int)Math.Ceiling(((double)PData.GetLength(0)) / ((double)PBatchSize));
            double[][][] batches = new double[numbatches][][];
            for(int i = 0;i < numbatches;i++)
            {
                batches[i] = new double[PBatchSize][];
                for(int j = 0;j < PBatchSize;j++)
                {
                    batches[i][j] = PData[(i * PBatchSize) + j];
                }
            }
            return batches;
        }
        #endregion

        #region ErrorObservation

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets or sets the error observers. </summary>
        ///
        /// <value> The error observers. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public List<IErrorObserver> ErrorObservers
        {
            get => errorobservers;
            set
            {
                if (value != null)
                {
                    errorobservers = value;
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Calculates the error. </summary>
        ///
        /// <exception cref="Exception">    Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="POriginal">        The original. </param>
        /// <param name="PReconstruction">  The reconstruction. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void CalculateError(double[] POriginal, double[] PReconstruction)
        {
            int originallength = POriginal.GetLength(0);
            if (originallength != PReconstruction.GetLength(0))
            {
                throw new AutoEncoderException("Tried to calculate error for different size lists.");
            }
            double error = 0;
            for (int i = 0; i < originallength; i++)
            {
                double temp = POriginal[i] - PReconstruction[i];
                error += temp * temp;
            }
            error /= originallength;
            SendError(error);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sends an error. </summary>
        ///
        /// <param name="PError">   The error. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void SendError(double PError)
        {
            foreach (var t in errorobservers)
            {
                t.CalculateError(PError);
            }
        }
        #endregion

        #region Save/Load

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Saves. </summary>
        ///
        /// <param name="PFilename">    The filename to load. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void Save(string PFilename)
        {
            TextWriter file = new StreamWriter(PFilename);
            learnrate?.Save(file);
            recognitionweights?.Save(file);
            generativeweights?.Save(file);
            file.WriteLine(numlayers);

            for (int i = 0; i < numlayers; i++)
            {
                if(layers[i].GetType() == typeof(RestrictedBoltzmannMachineGaussianLayer))
                {
                    file.WriteLine("RestrictedBoltzmannMachineGaussianLayer");
                }
                else if (layers[i].GetType() == typeof(RestrictedBoltzmannMachineBinaryLayer))
                {
                    file.WriteLine("RestrictedBoltzmannMachineBinaryLayer");
                }
                layers[i].Save(file);
            }
            file.WriteLine(pretraining);
            file.Close();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Loads. </summary>
        ///
        /// <param name="PFilename">    The filename to load. </param>
        ///
        /// <returns>   An Autoencoder. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static Autoencoder Load(string PFilename)
        {
            TextReader file = new StreamReader(PFilename);
            Autoencoder retval = new Autoencoder();
            retval.learnrate = new AutoencoderLearningRate();
            retval.learnrate.Load(file);
            retval.recognitionweights = new AutoencoderWeights();
            retval.recognitionweights.Load(file);
            retval.generativeweights = new AutoencoderWeights();
            retval.generativeweights.Load(file);
            retval.numlayers = int.Parse(file.ReadLine());
            retval.layers = new RestrictedBoltzmannMachineLayer[retval.numlayers];
            for (int i = 0; i < retval.numlayers; i++)
            {
                string type = file.ReadLine();
                if (type == "RestrictedBoltzmannMachineGaussianLayer")
                {
                    retval.layers[i] = new RestrictedBoltzmannMachineGaussianLayer();
                }
                else if (type == "RestrictedBoltzmannMachineBinaryLayer")
                {
                    retval.layers[i] = new RestrictedBoltzmannMachineBinaryLayer();
                }
                retval.layers[i].Load(file);
            }
            retval.pretraining = bool.Parse(file.ReadLine());
            retval.InitializeTrainingData();
            retval.errorobservers = new List<IErrorObserver>();
            file.Close();
            return retval;
        }
        #endregion
    }
}

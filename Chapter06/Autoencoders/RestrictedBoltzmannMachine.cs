
using System;

namespace Autoencoders
{
    /// <summary>   A restricted boltzmann machine. </summary>
    public class RestrictedBoltzmannMachine
    {
        /// <summary>   The visible layers. </summary>
        private RestrictedBoltzmannMachineLayer visibleLayers;
        /// <summary>   The hidden layers. </summary>
        private RestrictedBoltzmannMachineLayer hiddenLayers;
        /// <summary>   The weights. </summary>
        private RestrictedBoltzmannMachineWeightSet weights;
        /// <summary>   The learnrate. </summary>
        private RestrictedBoltzmannMachineLearningRate learnrate;
        /// <summary>   The trainingData. </summary>
        private TrainingData trainingData;
        /// <summary>   Number of visible layers. </summary>
        private int numVisibleLayers;
        /// <summary>   Number of hidden layers. </summary>
        private int numHiddenLayers;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the weights. </summary>
        ///
        /// <value> The weights. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public RestrictedBoltzmannMachineWeightSet Weights => weights;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets information describing the visible. </summary>
        ///
        /// <value> Information describing the visible. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double[] VisibleData
        {
            get
            {
                double[] retval = new double[numVisibleLayers];
                for (int i = 0; i < numVisibleLayers; i++)
                {
                    retval[i] = visibleLayers.GetState(i);
                }
                return retval;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the visible activity. </summary>
        ///
        /// <value> The visible activity. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double[] VisibleActivity
        {
            get
            {
                double[] retval = new double[numVisibleLayers];
                for (int i = 0; i < numVisibleLayers; i++)
                {
                    retval[i] = visibleLayers.GetActivity(i);
                }
                return retval;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets information describing the hidden. </summary>
        ///
        /// <value> Information describing the hidden. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double[] HiddenData
        {
            get
            {
                double[] retval = new double[numHiddenLayers];
                for (int i = 0; i < numHiddenLayers; i++)
                {
                    retval[i] = hiddenLayers.GetState(i);
                }
                return retval;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the hidden activity. </summary>
        ///
        /// <value> The hidden activity. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double[] HiddenActivity
        {
            get
            {
                double[] retval = new double[numHiddenLayers];
                for (int i = 0; i < numHiddenLayers; i++)
                {
                    retval[i] = hiddenLayers.GetActivity(i);
                }
                return retval;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the number of visibles. </summary>
        ///
        /// <value> The total number of visibles. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public int NumVisibles => numVisibleLayers;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the number of hiddens. </summary>
        ///
        /// <value> The total number of hiddens. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public int NumHiddens => numHiddenLayers;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets or sets the learn rate. </summary>
        ///
        /// <value> The learn rate. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public RestrictedBoltzmannMachineLearningRate LearnRate
        {
            get => learnrate;
            set => learnrate = value;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the Autoencoders.RestrictedBoltzmannMachine class.
        /// </summary>
        ///
        /// <param name="PVisibles">    The visibles. </param>
        /// <param name="PHiddens">     The hiddens. </param>
        /// <param name="PLearnRate">   The learn rate. </param>
        /// <param name="PWeightInit">  The weight initialize. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public RestrictedBoltzmannMachine(RestrictedBoltzmannMachineLayer PVisibles, RestrictedBoltzmannMachineLayer PHiddens, RestrictedBoltzmannMachineLearningRate PLearnRate, IWeightInitializer PWeightInit)
        {
            numVisibleLayers = PVisibles.Count;
            numHiddenLayers = PHiddens.Count; 
            InitLayers(PVisibles, PHiddens);
            InitWeights(PWeightInit);
            InitTrainingData();
            learnrate = PLearnRate;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the Autoencoders.RestrictedBoltzmannMachine class.
        /// </summary>
        ///
        /// <param name="PA">   The pa. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public RestrictedBoltzmannMachine(RestrictedBoltzmannMachine PA)
        {
            visibleLayers = (RestrictedBoltzmannMachineLayer)PA?.visibleLayers?.Clone();
            hiddenLayers = (RestrictedBoltzmannMachineLayer)PA?.hiddenLayers?.Clone();
            weights = (RestrictedBoltzmannMachineWeightSet)PA?.weights?.Clone();
            learnrate = PA.learnrate;
            trainingData = PA.trainingData;
            numVisibleLayers = PA.numVisibleLayers;
            numHiddenLayers = PA.numHiddenLayers;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes the layers. </summary>
        ///
        /// <exception cref="ArgumentNullException">    Thrown when one or more required arguments are
        ///                                             null. </exception>
        /// <exception cref="ArgumentException">        Thrown when one or more arguments have
        ///                                             unsupported or illegal values. </exception>
        ///
        /// <param name="PVisibles">    The visibles. </param>
        /// <param name="PHiddens">     The hiddens. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void InitLayers(RestrictedBoltzmannMachineLayer PVisibles, RestrictedBoltzmannMachineLayer PHiddens)
        {
            if (PVisibles == null)
            {
                throw new ArgumentNullException("You need a visible layer...");
            }
            if (PHiddens == null)
            {
                throw new ArgumentNullException("You need a hidden layer...");
            }
            if (PVisibles.Count <= 0)
            {
                throw new ArgumentException("You need at least one visible neuron...");
            }
            if (PHiddens.Count <= 0)
            {
                throw new ArgumentException("You need at least one hidden neuron...");
            }
            hiddenLayers = PHiddens;
            visibleLayers = PVisibles;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes the weights. </summary>
        ///
        /// <param name="PWeightInit">  The weight initialize. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void InitWeights(IWeightInitializer PWeightInit)
        {
            weights = new RestrictedBoltzmannMachineWeightSet(numVisibleLayers, numHiddenLayers, PWeightInit);
            for (int i = 0; i < numVisibleLayers; i++)
            {
                for (int j = 0; j < numHiddenLayers; j++)
                {
                    weights.SetWeight(i, j, Utility.NextGaussian(0,0.1));
                }
            }
        }
        /// <summary>   Initializes the training data. </summary>
        private void InitTrainingData()
        {
            trainingData = new TrainingData
            {
                posVisible = new double[numVisibleLayers]
            };
            Utility.SetArrayToZero(trainingData.posVisible);
            trainingData.posHidden = new double[numHiddenLayers];
            Utility.SetArrayToZero(trainingData.posHidden);
            trainingData.negVisible = new double[numVisibleLayers];
            Utility.SetArrayToZero(trainingData.negVisible);
            trainingData.negHidden = new double[numHiddenLayers];
            Utility.SetArrayToZero(trainingData.negHidden);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Calculates the error. </summary>
        ///
        /// <returns>   The calculated error. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double CalculateError()
        {
            double error = 0;
            for (int i = 0; i < numVisibleLayers; i++)
            {
                double temp = trainingData.posVisible[i] - trainingData.negVisible[i];
                error += temp * temp;
            }
            error /= numVisibleLayers;
            return error;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Compress the given p data. </summary>
        ///
        /// <param name="PData">    The data. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void Compress(double[] PData)
        {
            SetVisibleData(PData);
            UpdateHiddens();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Reconstructs this object. </summary>
        ///
        /// <param name="PData">    The data. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void Reconstruct(double[] PData)
        {
            SetHiddenData(PData);
            Reconstruct();
        }
        /// <summary>   Reconstructs this object. </summary>
        public void Reconstruct()
        {
            UpdateVisibles();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Trains the given p data. </summary>
        ///
        /// <exception cref="ArgumentNullException">    Thrown when one or more required arguments are
        ///                                             null. </exception>
        ///
        /// <param name="PBatchData">   Information describing the batch. </param>
        ///
        /// <returns>   A double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double Train(double[][] PBatchData)
        {
            if (PBatchData == null)
            {
                throw new ArgumentNullException("Bad training data!");
            } 
            
            double error = 0;
            trainingData.Zero();
            for (int i = 0; i < PBatchData.GetLength(0); i++)
            {
                SaveTrainingData(PBatchData[i]);
                error += CalculateError();
            }
            error /= PBatchData.GetLength(0);
            trainingData.Scalar(1 / PBatchData.GetLength(0));
            PerformTraining();
            return error;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Trains the given p data. </summary>
        ///
        /// <param name="PData">    The data. </param>
        ///
        /// <returns>   A double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double Train(double[] PData)
        {
            double[][] batch = new double[1][];
            batch[0] = PData;
            return Train(batch);
        }

        /// <summary>   Performs the training action. </summary>
        private void PerformTraining()
        {
            RestrictedBoltzmannMachineTrainer.Train(visibleLayers, hiddenLayers, trainingData, learnrate, weights);
            
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Saves a training data. </summary>
        ///
        /// <param name="PData">    The data. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void SaveTrainingData(double[] PData)
        {
            PositivePhase(PData);

            NegativePhase();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Positive phase. </summary>
        ///
        /// <param name="PData">    The data. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void PositivePhase(double[] PData)
        {
            SetVisibleData(PData);
            UpdateHiddens();
            Utility.AddArrays(trainingData.posVisible, VisibleActivity);
            Utility.AddArrays(trainingData.posHidden, HiddenActivity);
        }
        /// <summary>   Negative phase. </summary>
        private void NegativePhase()
        {
            UpdateVisibles();
            UpdateHiddens();
            Utility.AddArrays(trainingData.negVisible, VisibleActivity);
            Utility.AddArrays(trainingData.negHidden, HiddenActivity);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets visible data. </summary>
        ///
        /// <exception cref="ArgumentException">    Thrown when one or more arguments have unsupported or
        ///                                         illegal values. </exception>
        /// <exception cref="Exception">            Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="PData">    The data. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void SetVisibleData(double[] PData)
        {
            if (PData.GetLength(0) != numVisibleLayers)
            {
                throw new ArgumentException("Too little or too much initial data");
            }
            for (int i = 0; i < numVisibleLayers; i++)
            {
                visibleLayers.SetStateBypass(i, PData[i]);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets hidden data. </summary>
        ///
        /// <exception cref="ArgumentOutOfRangeException">  Thrown when one or more arguments are outside
        ///                                                 the required range. </exception>
        ///
        /// <param name="PData">    The data. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void SetHiddenData(double[] PData)
        {
            if (PData.GetLength(0) != numHiddenLayers)
            {
                throw new ArgumentOutOfRangeException("Too little or too much initial data");
            }
            for (int i = 0; i < numHiddenLayers; i++)
            {
                hiddenLayers.SetStateBypass(i, PData[i]);
            }
        }
        /// <summary>   Updates the hiddens. </summary>
        private void UpdateHiddens()
        {
            double input = 0;
            double[] states = visibleLayers.GetStates();
            for (int i = 0; i < numHiddenLayers; i++)
            {
                for (int j = 0; j < numVisibleLayers; j++)
                {
                    input += weights.GetWeight(j, i) * states[j];
                }
                hiddenLayers.SetState(i, input);
                input = 0;
            }
        }
        /// <summary>   Updates the visibles. </summary>
        private void UpdateVisibles()
        {
            double input = 0;
            double[] states = hiddenLayers.GetStates();
            for (int i = 0; i < numVisibleLayers; i++)
            {
                for (int j = 0; j < numHiddenLayers; j++)
                {
                    input += weights.GetWeight(i, j) * states[j];
                }
                visibleLayers.SetState(i,input);
                input = 0;
            }
        }
    }
}
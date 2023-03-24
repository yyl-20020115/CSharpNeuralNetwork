using System;
using System.IO;

namespace Autoencoders
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   A restricted boltzmann machine layer. </summary>
    ///
    /// <seealso cref="T:System.ICloneable"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public abstract class RestrictedBoltzmannMachineLayer: ICloneable
    {
        /// <summary>   The state. </summary>
        protected double[] state;
        /// <summary>   The bias. </summary>
        protected double[] bias;
        /// <summary>   The bias change. </summary>
        protected double[] biasChange;
        /// <summary>   The activity. </summary>
        protected double[] activity;
        /// <summary>   Number of neurons. </summary>
        protected int numNeurons = 0;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the Autoencoders.RestrictedBoltzmannMachineLayer class.
        /// </summary>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public RestrictedBoltzmannMachineLayer()
        {

        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the Autoencoders.RestrictedBoltzmannMachineLayer class.
        /// </summary>
        ///
        /// <exception cref="Exception">    Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="PSize">    The size. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public RestrictedBoltzmannMachineLayer(int PSize)
        {
            if (PSize <= 0)
            {
                throw new Exception("Can't have a layer without neurons!");
            }
            numNeurons = PSize;
            state = new double[numNeurons];
            bias = new double[numNeurons];
            biasChange = new double[numNeurons];
            activity = new double[numNeurons];
            for (int i = 0; i < PSize; i++)
            {
                state[i] = 0;
                bias[i] = 0;
                biasChange[i] = 0;
                activity[i] = 0;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets state bypass. </summary>
        ///
        /// <param name="PWhich">   The which. </param>
        /// <param name="PState">   The state. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void SetStateBypass(int PWhich, double PState)
        {
            CheckBounds(PWhich);
            state[PWhich] = PState;
            activity[PWhich] = PState;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets a state. </summary>
        ///
        /// <param name="PWhich">   The which. </param>
        ///
        /// <returns>   The state. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double GetState(int PWhich)
        {
            CheckBounds(PWhich);
            return state[PWhich];
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets the bias. </summary>
        ///
        /// <param name="PWhich">   The which. </param>
        /// <param name="PBias">    The bias. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void SetBias(int PWhich, double PBias)
        {
            CheckBounds(PWhich);
            bias[PWhich] = PBias;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the bias. </summary>
        ///
        /// <param name="PWhich">   The which. </param>
        ///
        /// <returns>   The bias. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double GetBias(int PWhich)
        {
            CheckBounds(PWhich);
            return bias[PWhich];
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets bias change. </summary>
        ///
        /// <param name="PWhich">       The which. </param>
        /// <param name="PBiasChange">  The bias change. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void SetBiasChange(int PWhich, double PBiasChange)
        {
            CheckBounds(PWhich);
            biasChange[PWhich] = PBiasChange;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets bias change. </summary>
        ///
        /// <param name="PWhich">   The which. </param>
        ///
        /// <returns>   The bias change. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double GetBiasChange(int PWhich)
        {
            CheckBounds(PWhich);
            return biasChange[PWhich];
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets an activity. </summary>
        ///
        /// <param name="PWhich">   The which. </param>
        ///
        /// <returns>   The activity. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double GetActivity(int PWhich)
        {
            CheckBounds(PWhich);
            return activity[PWhich];
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the states. </summary>
        ///
        /// <returns>   An array of double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double[] GetStates()
        {
            return (double[])state.Clone();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the activities. </summary>
        ///
        /// <returns>   An array of double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double[] GetActivities()
        {
            return (double[])activity.Clone();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the number of.  </summary>
        ///
        /// <value> The count. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public int Count => numNeurons;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Check bounds. </summary>
        ///
        /// <exception cref="Exception">    Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="PWhich">   The which. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void CheckBounds(int PWhich)
        {
            if (PWhich < 0 || PWhich >= numNeurons)
            {
                throw new Exception("Index out of bounds!!!!! GOOFBALL");
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets a state. </summary>
        ///
        /// <param name="PWhich">   The which. </param>
        /// <param name="PInput">   The input. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public abstract void SetState(int PWhich, double PInput);

        #region ICloneable Members

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Creates a new object that is a copy of the current instance. </summary>
        ///
        /// <returns>   A new object that is a copy of this instance. </returns>
        ///
        /// <seealso cref="M:System.ICloneable.Clone()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public abstract object Clone();

        #endregion

        #region Save/Load

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Saves the given p file. </summary>
        ///
        /// <param name="PFile">    The file to load. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal void Save(TextWriter PFile)
        {
            PFile.WriteLine(numNeurons);
            Utility.SaveArray(bias, PFile);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Loads the given p file. </summary>
        ///
        /// <param name="PFile">    The file to load. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal void Load(TextReader PFile)
        {
            numNeurons = int.Parse(PFile.ReadLine());
            bias = Utility.LoadArray(PFile);
            biasChange = new double[numNeurons];
            activity = new double[numNeurons];
            state = new double[numNeurons];
            for (int i = 0; i < numNeurons; i++)
            {
                biasChange[i] = 0;
                activity[i] = 0;
                state[i] = 0;
            }
        }
        #endregion
    }
}

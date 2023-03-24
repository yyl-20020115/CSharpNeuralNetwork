using System;
using System.IO;

namespace Autoencoders
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   A restricted boltzmann machine weight set. </summary>
    ///
    /// <seealso cref="T:System.ICloneable"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class RestrictedBoltzmannMachineWeightSet: ICloneable
    {
        /// <summary>   Size of the pre. </summary>
        private int preSize;
        /// <summary>   Size of the post. </summary>
        private int postSize;
        /// <summary>   The weights. </summary>
        private double[][] weights;
        /// <summary>   The weight changes. </summary>
        private double[][] weightChanges;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Prevents a default instance of the Autoencoders.RestrictedBoltzmannMachineWeightSet class
        /// from being created.
        /// </summary>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private RestrictedBoltzmannMachineWeightSet()
        {

        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the Autoencoders.RestrictedBoltzmannMachineWeightSet class.
        /// </summary>
        ///
        /// <param name="PPreSynapticLayerSize">    Size of the pre synaptic layer. </param>
        /// <param name="PPostSynapticLayerSize">   Size of the post synaptic layer. </param>
        /// <param name="PWeightInit">              The weight initialize. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public RestrictedBoltzmannMachineWeightSet(int PPreSynapticLayerSize, int PPostSynapticLayerSize, IWeightInitializer PWeightInit)
        {
            preSize = PPreSynapticLayerSize;
            postSize = PPostSynapticLayerSize;
            weights = new double[preSize][];
            weightChanges = new double[preSize][];
            for (int i = 0; i < preSize; i++)
            {
                weights[i] = new double[postSize];
                weightChanges[i] = new double[postSize];
                Utility.SetArrayToZero(weightChanges[i]);
                for (int j = 0; j < postSize; j++)
                {
                    weights[i][j] = PWeightInit.InitializeWeight();
                }
            }
        }

        #region WeightModification

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Modify weight. </summary>
        ///
        /// <param name="PPre">     The pre. </param>
        /// <param name="PPost">    The post. </param>
        /// <param name="PAmount">  The amount. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void ModifyWeight(int PPre, int PPost, double PAmount)
        {
            CheckSynapseExists(PPre, PPost);
            weightChanges[PPre][PPost] = PAmount;
            weights[PPre][PPost] += PAmount;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets a weight. </summary>
        ///
        /// <param name="PPre">     The pre. </param>
        /// <param name="PPost">    The post. </param>
        /// <param name="PValue">   The value. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void SetWeight(int PPre, int PPost, double PValue)
        {
            CheckSynapseExists(PPre, PPost);
            weightChanges[PPre][PPost] = PValue - weights[PPre][PPost];
            weights[PPre][PPost] = PValue;
        }
        #endregion

        #region Accessors

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets a weight. </summary>
        ///
        /// <param name="PPre">     The pre. </param>
        /// <param name="PPost">    The post. </param>
        ///
        /// <returns>   The weight. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double GetWeight(int PPre, int PPost)
        {
            CheckSynapseExists(PPre, PPost);
            return weights[PPre][PPost];
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets weight change. </summary>
        ///
        /// <param name="PPre">     The pre. </param>
        /// <param name="PPost">    The post. </param>
        ///
        /// <returns>   The weight change. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public double GetWeightChange(int PPre, int PPost)
        {
            CheckSynapseExists(PPre, PPost);
            return weightChanges[PPre][PPost];
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the size of the pre synaptic layer. </summary>
        ///
        /// <value> The size of the pre synaptic layer. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public int PreSynapticLayerSize => preSize;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the size of the post synaptic layer. </summary>
        ///
        /// <value> The size of the post synaptic layer. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public int PostSynapticLayerSize => postSize;

        #endregion

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Queries if a given check synapse exists. </summary>
        ///
        /// <param name="PPre">     The pre. </param>
        /// <param name="PPost">    The post. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void CheckSynapseExists(int PPre, int PPost)
        {
            Utility.WithinBounds("Pre-synaptic weight index out of bounds! K.O.", PPre, preSize);
            Utility.WithinBounds("Post-synaptic weight index out of bounds! K.O.", PPost, postSize);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Creates a new object that is a copy of the current instance. </summary>
        ///
        /// <returns>   A new object that is a copy of this instance. </returns>
        ///
        /// <seealso cref="M:System.ICloneable.Clone()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public object Clone()
        {
            RestrictedBoltzmannMachineWeightSet newweights = new RestrictedBoltzmannMachineWeightSet(preSize, postSize, new ZeroWeightInitializer());
            for (int i = 0; i < preSize; i++)
            {
                for (int j = 0; j < postSize; j++)
                {
                    newweights.SetWeight(i, j, weights[i][j]);
                }
            }
            return newweights;
        }

        #region Save/Load

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Saves the given p file. </summary>
        ///
        /// <param name="PFile">    The file to load. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal void Save(TextWriter PFile)
        {
            PFile.WriteLine(preSize);
            for (int i = 0; i < preSize; i++)
            {
                Utility.SaveArray(weights[i], PFile);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Loads the given p file. </summary>
        ///
        /// <param name="PFile">    The file to load. </param>
        ///
        /// <returns>   A RestrictedBoltzmannMachineWeightSet. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal static RestrictedBoltzmannMachineWeightSet Load(TextReader PFile)
        {
            RestrictedBoltzmannMachineWeightSet retval = new RestrictedBoltzmannMachineWeightSet
            {
                preSize = int.Parse(PFile.ReadLine())
            };
            retval.weights = new double[retval.preSize][];
            retval.weightChanges = new double[retval.preSize][];
            for (int i = 0; i < retval.preSize; i++)
            {
                retval.weights[i] = Utility.LoadArray(PFile);
            }
            retval.postSize = retval.weights[0].GetLength(0);
            for (int i = 0; i < retval.preSize; i++)
            {
                retval.weightChanges[i] = new double[retval.postSize];
            }
            return retval;
        }
        #endregion
    }
}

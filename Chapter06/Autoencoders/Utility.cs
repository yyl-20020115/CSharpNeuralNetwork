using System;
using System.IO;

namespace Autoencoders
{
    /// <summary>   An utility. </summary>
    public static class Utility
    {
        #region ArrayStuff

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Adds the arrays to 'PB'. </summary>
        ///
        /// <exception cref="Exception">    Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="PA">   The pa. </param>
        /// <param name="PB">   The pb. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static void AddArrays(double[] PA, double[] PB)
        {
            if (PA.GetLength(0) != PB.GetLength(0))
            {
                throw new ArgumentException("Array sizes cannot be different");
            }
            int arraylength = PA.GetLength(0);
            for (int i = 0; i < arraylength; i++)
            {
                PA[i] += PB[i];
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Scale array. </summary>
        ///
        /// <param name="PA">       The pa. </param>
        /// <param name="PScalar">  The scalar. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static void ScaleArray(double[] PA, double PScalar)
        {
            int arraylength = PA.GetLength(0);
            for (int i = 0; i < arraylength; i++)
            {
                PA[i] *= PScalar;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets array to zero. </summary>
        ///
        /// <param name="PA">   The pa. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static void SetArrayToZero(double[] PA)
        {
            int arraylength = PA.GetLength(0);
            for (int i = 0; i < arraylength; i++)
            {
                PA[i] = 0;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets array to zero. </summary>
        ///
        /// <param name="PA">   The pa. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static void SetArrayToZero(double[,] PA)
        {
            int arraylengtha = PA.GetLength(0);
            int arraylengthb = PA.GetLength(1);
            for (int i = 0; i < arraylengtha; i++)
            {
                for (int j = 0; j < arraylengthb; j++)
                {
                    PA[i,j] = 0;
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Saves an array. </summary>
        ///
        /// <param name="PA">       The pa. </param>
        /// <param name="PFile">    The file. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static void SaveArray(double[] PA, TextWriter PFile)
        {
            PFile.WriteLine(PA.GetLength(0));
            for (int i = 0; i < PA.GetLength(0); i++)
            {
                PFile.WriteLine(PA[i]);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Loads an array. </summary>
        ///
        /// <param name="PFile">    The file. </param>
        ///
        /// <returns>   An array of double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static double[] LoadArray(TextReader PFile)
        {
            int length = int.Parse(PFile.ReadLine());
            double[] retval = new double[length];
            for (int i = 0; i < length; i++)
            {
                retval[i] = double.Parse(PFile.ReadLine());
            }
            return retval;
        }
        #endregion

        #region Random
        /// <summary>   The random. </summary>
        static System.Random rand = new Random();        
        #region GaussianRandom
        /// <summary>   The next. </summary>
        static double next = 0;
        /// <summary>   True to nextset. </summary>
        static bool nextset = false;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Next gaussian. </summary>
        ///
        /// <param name="PMean">    The mean. </param>
        /// <param name="PStdDev">  The standard development. </param>
        ///
        /// <returns>   A double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static public double NextGaussian(double PMean, double PStdDev)
        {
            if (nextset)
            {
                nextset = false;
                return next;
            }

            double x1, x2, w;
            do
            {
                x1 = 2.0 * rand.NextDouble() - 1.0;
                x2 = 2.0 * rand.NextDouble() - 1.0;
                w = x1 * x1 + x2 * x2;
            } 
            while (w >= 1.0);

            nextset = true;
            next = (PMean + (x2 * PStdDev));
            return (PMean + (x1 * PStdDev));
        }
        #endregion

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Next double. </summary>
        ///
        /// <returns>   A double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static double NextDouble()
        {
            return rand.NextDouble();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Next double. </summary>
        ///
        /// <param name="PMax"> The maximum. </param>
        ///
        /// <returns>   A double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static double NextDouble(int PMax)
        {
            return rand.NextDouble() * PMax;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Next double. </summary>
        ///
        /// <param name="PMin"> The minimum. </param>
        /// <param name="PMax"> The maximum. </param>
        ///
        /// <returns>   A double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static double NextDouble(int PMin, int PMax)
        {
            return rand.NextDouble() * (PMax - PMin) + PMin;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Next int. </summary>
        ///
        /// <returns>   An int. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static int NextInt()
        {
            return rand.Next();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Next int. </summary>
        ///
        /// <param name="PMax"> The maximum. </param>
        ///
        /// <returns>   An int. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static int NextInt(int PMax)
        {
            return rand.Next(PMax);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Next int. </summary>
        ///
        /// <param name="PMin"> The minimum. </param>
        /// <param name="PMax"> The maximum. </param>
        ///
        /// <returns>   An int. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static int NextInt(int PMin, int PMax)
        {
            return rand.Next(PMin, PMax);
        }
        #endregion

        #region ErrorHandling

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Within bounds. </summary>
        ///
        /// <param name="PErrorMessage">    Message describing the error. </param>
        /// <param name="PValue">           The value. </param>
        /// <param name="PUpper">           The upper. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static void WithinBounds(string PErrorMessage, int PValue, int PUpper)
        {
            WithinBounds(PErrorMessage,PValue, 0, PUpper);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Within bounds. </summary>
        ///
        /// <param name="PValue">   The value. </param>
        /// <param name="PUpper">   The upper. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static void WithinBounds(int PValue, int PUpper)
        {
            WithinBounds("Index out of bounds!" ,PValue, 0, PUpper);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Within bounds. </summary>
        ///
        /// <param name="PValue">   The value. </param>
        /// <param name="PLower">   The lower. </param>
        /// <param name="PUpper">   The upper. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static void WithinBounds(int PValue,int PLower, int PUpper)
        {
            WithinBounds("Index out of bounds!" ,PValue, PLower, PUpper);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Within bounds. </summary>
        ///
        /// <exception cref="Exception">    Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="PErrorMessage">    Message describing the error. </param>
        /// <param name="PValue">           The value. </param>
        /// <param name="PLower">           The lower. </param>
        /// <param name="PUpper">           The upper. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static void WithinBounds(string PErrorMessage, int PValue, int PLower, int PUpper)
        {
            if(PValue < PLower || PValue >= PUpper)
            {
                throw new ArgumentOutOfRangeException(PErrorMessage);
            }
        }
        #endregion
    }
}

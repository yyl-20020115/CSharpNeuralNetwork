#region License
/*
    Autoencoders library.
    Copyright (C) 2007 Thomas Benjamin Thompson

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
*/
#endregion
using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace NeuralNetwork
{
    public static class Utility
    {
        #region ArrayStuff
        public static double[][] TransposeArray(double[][] PA)
        {
            int firstsize = PA.GetLength(0);
            int secondsize = PA[0].GetLength(0);
            double[][] retval = new double[secondsize][];
            for (int i = 0; i < secondsize; i++)
            {
                retval[i] = new double[firstsize];
                for (int j = 0; j < firstsize; j++)
                {
                    retval[i][j] = PA[j][i];
                }
            }
            return retval;
        }
        public static double InnerProduct(double[] PA, double[] PB)
        {
            double retval = 0;
            int arraylength = PA.GetLength(0); 
            for (int i = 0; i < arraylength; i++)
            {
                retval = PA[i] * PB[i];
            }
            return retval;
        }
        public static void CopyArraySizeInitZero(out double[] PA, double[] PB)
        {
            PA = new double[PB.GetLength(0)];
            for (int i = 0; i < PB.GetLength(0); i++)
            {
                PA[i] = 0;
            }
        }
        public static void CopyArraySizeInitZero(out double[][] PA, double[][] PB)
        {
            PA = new double[PB.GetLength(0)][];
            for (int i = 0; i < PB.GetLength(0); i++)
            {
                CopyArraySizeInitZero(out PA[i], PB[i]);
            }
        }
        public static void InitDefaultArray(out double[] PA, int PSize, double PValue)
        {
            PA = new double[PSize];
            for (int i = 0; i < PSize; i++)
            {
                PA[i] = PValue;
            }
        }
        public static void InitDefaultArray(out double[][] PA, int PSize1, int PSize2, double PValue)
        {
            PA = new double[PSize1][];
            for (int i = 0; i < PSize1; i++)
            {
                InitDefaultArray(out PA[i], PSize2, PValue);
            }
        }
        public static void InitDefaultArray(out double[][][] PA, int PSize1, int PSize2, int PSize3, double PValue)
        {
            PA = new double[PSize1][][];
            for (int i = 0; i < PSize1; i++)
            {
                InitDefaultArray(out PA[i],PSize2, PSize3, PValue);
            }
        }
        public static void AddArrays(double[] PA, double[] PB)
        {
            int arraylength = PA.GetLength(0);
            if (arraylength != PB.GetLength(0))
            {
                throw new Exception("YOU CAN'T ADD ARRAYS OF DIFFERENT SIZES... GOSH!");
            }
            for (int i = 0; i < arraylength; i++)
            {
                PA[i] += PB[i];
            }
        }
        public static void AddArrays(double[][] PA, double[][] PB)
        {
            int arraylength = PA.GetLength(0);
            if (arraylength != PB.GetLength(0))
            {
                throw new Exception("YOU CAN'T ADD ARRAYS OF DIFFERENT SIZES... GOSH!");
            }
            for (int i = 0; i < arraylength; i++)
            {
                Utility.AddArrays(PA[i],PB[i]);
            }
        }
        public static void AddArrays(double[][][] PA, double[][][] PB)
        {
            int arraylength = PA.GetLength(0);
            if (arraylength != PB.GetLength(0))
            {
                throw new Exception("YOU CAN'T ADD ARRAYS OF DIFFERENT SIZES... GOSH!");
            }
            for (int i = 0; i < arraylength; i++)
            {
                Utility.AddArrays(PA[i], PB[i]);
            }
        }
        public static void ScaleArray(double[] PA, double PScalar)
        {
            int arraylength = PA.GetLength(0);
            for (int i = 0; i < arraylength; i++)
            {
                PA[i] *= PScalar;
            }
        }
        public static void ScaleArray(double[][] PA, double PScalar)
        {
            int arraylength = PA.GetLength(0);
            for (int i = 0; i < arraylength; i++)
            {
                ScaleArray(PA[i],PScalar);
            }
        }
        public static void ScaleArray(double[][][] PA, double PScalar)
        {
            int arraylength = PA.GetLength(0);
            for (int i = 0; i < arraylength; i++)
            {
                ScaleArray(PA[i], PScalar);
            }
        }
        public static void ZeroArray(double[] PA)
        {
            int arraylength = PA.GetLength(0);
            for (int i = 0; i < arraylength; i++)
            {
                PA[i] = 0;
            }
        }
        public static void ZeroArray(double[][] PA)
        {
            int arraylength = PA.GetLength(0);
            for (int i = 0; i < arraylength; i++)
            {
                ZeroArray(PA[i]);
            }
        }
        public static void ZeroArray(double[][][] PA)
        {
            int arraylength = PA.GetLength(0);
            for (int i = 0; i < arraylength; i++)
            {
                ZeroArray(PA[i]);
            }
        }
        public static void ResetArray(double[] PA, double PValue)
        {
            int arraylength = PA.GetLength(0);
            for (int i = 0; i < arraylength; i++)
            {
                PA[i] = PValue;
            }
        }
        public static void ResetArray(double[][] PA, double PValue)
        {
            int arraylength = PA.GetLength(0);
            for (int i = 0; i < arraylength; i++)
            {
                ResetArray(PA[i], PValue);
            }
        }
        public static void SaveArray(double[] PA, TextWriter PFile)
        {
            PFile.WriteLine(PA.GetLength(0));
            for (int i = 0; i < PA.GetLength(0); i++)
            {
                PFile.WriteLine(PA[i]);
            }
        }
        public static void SaveArray(double[][] PA, TextWriter PFile)
        {
            PFile.WriteLine(PA.GetLength(0));
            for (int i = 0; i < PA.GetLength(0); i++)
            {
                SaveArray(PA[i], PFile);
            }
        }
        public static double[] LoadArray(TextReader PFile)
        {
            int length = int.Parse(PFile.ReadLine());
            double[] retval = new double[length];
            for (int i = 0; i < length; i++)
            {
                string stuff = PFile.ReadLine();
                if (stuff == null)
                {
                    retval[i] = 0;
                }
                else
                {
                    retval[i] = double.Parse(stuff);
                }
            }
            return retval;
        }
        public static double[][] LoadArray2D(TextReader PFile)
        {
            int length = int.Parse(PFile.ReadLine());
            double[][] retval = new double[length][];
            for (int i = 0; i < length; i++)
            {
                retval[i] = LoadArray(PFile);
            }
            return retval;
        }
        #endregion

        #region Random
        static System.Random rand = new System.Random();        
        #region GaussianRandom
        static double next = 0;
        static bool nextset = false;
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
            } while (w >= 1.0);
            w = Math.Sqrt((-2.0 * Math.Log(w, Math.E)) / w);
            nextset = true;
            next = (PMean + (x2 * PStdDev));
            return (PMean + (x1 * PStdDev));
        }
        #endregion

        public static double NextDouble()
        {
            return rand.NextDouble();
        }
        public static double NextDouble(int PMax)
        {
            return (rand.NextDouble() * PMax);
        }
        public static double NextDouble(int PMin, int PMax)
        {
            return ((rand.NextDouble() * (PMax - PMin)) + PMin);
        }
        public static int NextInt()
        {
            return rand.Next();
        }
        public static int NextInt(int PMax)
        {
            return rand.Next(PMax);
        }
        public static int NextInt(int PMin, int PMax)
        {
            return rand.Next(PMin, PMax);
        }
        #endregion

        #region ErrorHandling
        public static void WithinBounds(string PErrorMessage, int PValue, int PUpper)
        {
            WithinBounds(PErrorMessage,PValue, 0, PUpper);
        }
        public static void WithinBounds(int PValue, int PUpper)
        {
            WithinBounds("Index out of bounds!" ,PValue, 0, PUpper);
        }
        public static void WithinBounds(int PValue,int PLower, int PUpper)
        {
            WithinBounds("Index out of bounds!" ,PValue, PLower, PUpper);
        }
        public static void WithinBounds(string PErrorMessage, int PValue, int PLower, int PUpper)
        {
            //if it's not within the bounds then throw the exception
            if(!(PValue >= PLower && PValue < PUpper))
            {
                throw new Exception(PErrorMessage);
            }
        }
        #endregion

        public static double[][][] CreateBatches(double[][] PData, int PBatchSize)
        {
            int numbatches = (int)Math.Ceiling(((double)PData.GetLength(0)) / ((double)PBatchSize));
            double[][][] batches = new double[numbatches][][];
            for (int i = 0; i < numbatches; i++)
            {
                batches[i] = new double[PBatchSize][];
                for (int j = 0; j < PBatchSize; j++)
                {
                    batches[i][j] = PData[(i * PBatchSize) + j];
                }
            }
            return batches;
        }

        public static void SwitchVars(ref double PA, ref double PB)
        {
            double temp = PA;
            PA = PB;
            PB = temp;
        }
        public static void SwitchVars(ref int PA, ref int PB)
        {
            int temp = PA;
            PA = PB;
            PB = temp;
        }
    }
}

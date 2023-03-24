﻿using System;
using System.Linq;

namespace SharpLearning.CrossValidation.TrainingTestSplitters
{
    /// <summary>
    /// Container for training and test indices
    /// </summary>
    public sealed class TrainingTestIndexSplit : IEquatable<TrainingTestIndexSplit>
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly int[] TrainingIndices;

        /// <summary>
        /// 
        /// </summary>
        public readonly int[] TestIndices;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="trainingIndices"></param>
        /// <param name="testIndices"></param>
        public TrainingTestIndexSplit(int[] trainingIndices, int[] testIndices)
        {
            if (trainingIndices == null) { throw new ArgumentNullException("trainingIndices"); }
            if (testIndices == null) { throw new ArgumentNullException("validationIndices"); }
            TrainingIndices = trainingIndices;
            TestIndices = testIndices;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool Equals(TrainingTestIndexSplit other)
        {
            if (!TrainingIndices.SequenceEqual(other.TrainingIndices)) { return false; }
            if (!TestIndices.SequenceEqual(other.TestIndices)) { return false; }

            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            TrainingTestIndexSplit other = obj as TrainingTestIndexSplit;
            if (other != null && Equals(other))
            {
                return true;
            }

            return false;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            unchecked // Overflow is fine, just wrap
            {
                int hash = 17;
                hash = hash * 23 + TrainingIndices.GetHashCode();
                hash = hash * 23 + TestIndices.GetHashCode();

                return hash;
            }
        }
    }
}

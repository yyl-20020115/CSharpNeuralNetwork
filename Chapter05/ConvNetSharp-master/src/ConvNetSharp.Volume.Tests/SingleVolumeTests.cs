﻿using System.Linq;
using ConvNetSharp.Volume.Single;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Volume.Tests
{
    [TestClass]
    public class SingleVolumeTests : VolumeTests<float>
    {
        protected override Volume<float> NewVolume(double[] values, Shape shape)
        {
            var converted = values.Select(i => (float) i).ToArray();
            return BuilderInstance.Volume.From(converted, shape);
        }
    }
}
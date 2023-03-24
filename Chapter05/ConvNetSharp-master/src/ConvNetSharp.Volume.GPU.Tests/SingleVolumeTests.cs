﻿using System.Linq;
using ConvNetSharp.Volume.GPU.Single;
using ConvNetSharp.Volume.Tests;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ConvNetSharp.Volume.GPU.Tests
{
    [TestClass]
    public class SingleVolumeTests : VolumeTests<float>
    {
        public SingleVolumeTests()
        {
            BuilderInstance<float>.Volume = new VolumeBuilder();
        }
        protected override Volume<float> NewVolume(double[] values, Shape shape)
        {
            var converted = values.Select(i => (float)i).ToArray();
            return BuilderInstance.Volume.From(converted, shape);
        }
    }
}
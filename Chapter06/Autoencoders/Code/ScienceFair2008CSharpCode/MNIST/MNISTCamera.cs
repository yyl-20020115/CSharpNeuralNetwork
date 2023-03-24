using System;
using System.Collections.Generic;
using System.Text;
using Sharp3D.Math.Core;
using Totem.CameraSystem;

namespace ScienceFair2008
{
    public class MNISTCamera: ICamera
    {
        public MNISTCamera(Vector3F PPosition, Vector3F PView)
        {
            position = PPosition;
            look = PView;
            up = new Vector3F(0.0f, 1.0f, 0.0f);
        }
        public override void Update(double ElapsedTime)
        {

        }
    }
}

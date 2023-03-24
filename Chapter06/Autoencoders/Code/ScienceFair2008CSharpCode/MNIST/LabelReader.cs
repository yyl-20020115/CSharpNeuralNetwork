using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace ScienceFair2008
{
    public static class LabelReader
    {
        public static int[] ReadLabels(string PFileName, bool PTraining)
        {
            if (!File.Exists(PFileName))
            {
                throw new Exception("Blaaaaaah....");
            }
            int[] retval;
            BinaryReader file = new BinaryReader(File.Open(PFileName, FileMode.Open));
            file.ReadByte();
            file.ReadByte();
            file.ReadByte();
            file.ReadByte();
            file.ReadByte();
            file.ReadByte();
            file.ReadByte();
            file.ReadByte();
            int numitems = 10000;
            if (PTraining)
            {
                numitems = 60000;
            }
            retval = new int[numitems];
            for (int i = 0; i < numitems; i++)
            {
                retval[i] = file.ReadByte();
            }
            file.Close();
            return retval;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Text;
using NNBase;
using FlickrNet;
using System.Drawing;
using System.Net;
using System.Drawing.Imaging;

namespace ScienceFair2008
{
    public enum ImageType
    {
        SquareThumbnail,
        Thumbnail,
        Small,
        Medium,
        Large,
        Original
    }
    public class FlickrInputProvider:INNInputProvider
    {
        private static Random Rand = new Random();

        private const string apikey = "842bed08712f3f123e173f837dd2a5af";
        private const string secret = "ee08b92d62b6fb2a";
        private Flickr flickr;

        private ImageType imagedownloadtype = ImageType.SquareThumbnail;
        private Photos[]  photocollections;
        private int       numcollections;
        private string[]  tags;
        private int       imagesizex;
        private int       imagesizey;
        private int       whichcollection = 0;
        private int[]     index;
        private int       numperpage = 100;
        private int[]     whichpage;
        private bool      sequential = true;
        private double[]  curphotodata;
        private double[]  curoutputs;
        private int       cursection;

        public int ImageSizeX
        {
            get { return imagesizex; }
        }
        public int ImageSizeY
        {
            get { return imagesizey; }
        }
        public int NumPixelsPerImage
        {
            get
            {
                return imagesizex * imagesizey;
            }
        }
        public int NumCollections
        {
            get
            {
                return numcollections;
            }
        }
        public int CurrentCollection
        {
            get
            {
                return whichcollection;
            }
        }
        public int PageSize
        {
            get
            {
                return numperpage;
            }
        }
        public bool Sequential
        {
            get { return sequential; }
            set { sequential = value; }
        }
        public double[] CurrentOutputs
        {
            get { return curphotodata; }
        }

        public FlickrInputProvider(string[] PTags, int PNumPerPage)
        {
            tags = (string[])PTags.Clone();
            numcollections = tags.GetLength(0);
            index = new int[numcollections];
            whichpage = new int[numcollections];
            numperpage = PNumPerPage;
            flickr = new Flickr(apikey, secret);
            photocollections = new Photos[numcollections];

            LoadAllCollections(0);

            SetPicture(photocollections[whichcollection]);
        }

        public void ResetPageAndIndex()
        {
            LoadAllCollections(0);
            whichcollection = 0;
        }

        private void LoadAllCollections(int PWhichPage)
        {
            for (int i = 0; i < numcollections; i++)
            {
                LoadOneCollection(i, PWhichPage);
            }
        }
        private void LoadOneCollection(int PWhichCollection, int PWhichPage)
        {
            index[PWhichCollection] = 0;
            whichpage[PWhichCollection] = PWhichPage;
            PhotoSearchOptions searchoptions = SearchOptionsForCollection(PWhichCollection, PWhichPage);
            photocollections[PWhichCollection] = flickr.PhotosSearch(searchoptions);
        }

        public void NextPage(int PWhichCollection)
        {
            int nextpage = whichpage[PWhichCollection] + 1;
            if (nextpage == photocollections[PWhichCollection].TotalPages)
            {
                nextpage = 0;
            }
            LoadOneCollection(PWhichCollection, nextpage);
        }

        private PhotoSearchOptions SearchOptionsForCollection(int PWhichCollection, int PWhichPage)
        {
            PhotoSearchOptions searchoptions = new PhotoSearchOptions();
            searchoptions.Tags = tags[PWhichCollection];
            searchoptions.AddLicense(1);
            searchoptions.AddLicense(2);
            searchoptions.AddLicense(3);
            searchoptions.AddLicense(4);
            searchoptions.AddLicense(5);
            searchoptions.AddLicense(6);
            searchoptions.SortOrder = PhotoSearchSortOrder.InterestingnessDesc;
            searchoptions.PerPage = numperpage;
            searchoptions.Page = PWhichPage;
            return searchoptions;
        }

        private void NextCollection()
        {
            whichcollection++;
            if (whichcollection >= numcollections)
            {
                whichcollection = 0;
            }
        }
        private void SetPicture(Photos PPhotoCollection)
        {
            int curindex = GetPhotoIndex();

            Photo currentphoto = PPhotoCollection.PhotoCollection[curindex];

            curphotodata = PictureData(DownloadPhoto(currentphoto));

            IncrementIndex();

            cursection = 0;
        }
        private void SetSection()
        {
            cursection++;
            if (cursection == 9)
            {
                whichcollection = Rand.Next() % numcollections;
                SetPicture(photocollections[whichcollection]);
            }
            curoutputs = new double[imagesizex * imagesizey];
            int xpos = (int)Math.Floor(((double)cursection) / 3);
            int ypos = cursection % 3;
            for (int i = 0; i < imagesizex; i++)
            {
                for (int j = 0; j < imagesizey; j++)
                {
                    curoutputs[i * imagesizey + j] = GetPhotoDataPoint((xpos * 25) + i, (ypos * 25) + j);
                }
            }
        }

        private double GetPhotoDataPoint(int PX, int PY)
        {
            double val = curphotodata[PX * 75 + PY];
            return val;
        }

        private int GetPhotoIndex()
        {
            int retval = 0;
            if (sequential)
            {
                retval = index[whichcollection] % numperpage;
            }
            else
            {
                retval = Rand.Next() % numperpage;
            }
            return retval;
        }
        private Bitmap DownloadPhoto(Photo PPhoto)
        {
            WebRequest request = WebRequest.Create(GetUrl(PPhoto));
            Bitmap retval = new Bitmap(request.GetResponse().GetResponseStream());
            imagesizex = retval.Width / 3;
            imagesizey = retval.Height / 3;
            return retval;
        }
        private string GetUrl(Photo PCurrentPhoto)
        {
            if (imagedownloadtype == ImageType.Large)
            {
                return PCurrentPhoto.LargeUrl;
            }
            else if (imagedownloadtype == ImageType.Medium)
            {
                return PCurrentPhoto.MediumUrl;
            }
            else if (imagedownloadtype == ImageType.Small)
            {
                return PCurrentPhoto.SmallUrl;
            }
            else if (imagedownloadtype == ImageType.Original)
            {
                return PCurrentPhoto.OriginalUrl;
            }
            else if (imagedownloadtype == ImageType.Thumbnail)
            {
                return PCurrentPhoto.ThumbnailUrl;
            }
            else if (imagedownloadtype == ImageType.SquareThumbnail)
            {
                return PCurrentPhoto.SquareThumbnailUrl;
            }
            return PCurrentPhoto.SquareThumbnailUrl;
        }
        private double[] PictureData(Bitmap PImage)
        {
            double[] data = new double[25 * 25 * 9];

            BitmapData bmd = PImage.LockBits(new Rectangle(0, 0, 75, 75),
                                             ImageLockMode.ReadOnly, PImage.PixelFormat);
            unsafe
            {
                byte* imgPtr = (byte*)bmd.Scan0;
                for (int i = 0; i < bmd.Width; i++)
                {
                    for (int j = 0; j < bmd.Height; j++)
                    {
                        double red = (((double)1) / ((double)255)) * ((double)(*imgPtr));
                        imgPtr++;
                        double green = (((double)1) / ((double)255)) * ((double)(*imgPtr));
                        imgPtr++;
                        double blue = (((double)1) / ((double)255)) * ((double)(*imgPtr)); 
                        imgPtr++;
                        data[j * 75 + i] = (red + green + blue) / 3;
                    }
                    imgPtr += bmd.Stride - (bmd.Width * 3);
                }
            }
            PImage.UnlockBits(bmd);
            return data;
        }
        private void IncrementIndex()
        {
            if (sequential)
            {
                index[whichcollection]++;
                if (index[whichcollection] == numperpage)
                {
                    NextPage(whichcollection);
                }
            }
        }


        public double[] NextInputCase()
        {
            SetSection();
            return curoutputs;
        }
        public double[] DesiredOutput()
        {
            return curoutputs;
        }
        public void Rewind(int PAmount)
        {
            for (int i = 0; i < numcollections; i++)
            {
                index[i] -= PAmount;
                if (index[i] < 0)
                {
                    index[i] = 0;
                }
            }
        }
        public void FastForward(int PAmount)
        {
            throw new Exception("The method or operation is not implemented.");
        }
    }
}

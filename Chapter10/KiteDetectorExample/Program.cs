using System;
using System.Collections.Generic;
using System.Linq;

namespace KiteDetectorExample
{
    using System.Configuration;
    using System.Drawing;
    using System.IO;
    using System.Net;
    using System.Reflection;
    using EnsureThat;
    using ExampleCommon;
    using ICSharpCode.SharpZipLib.GZip;
    using ICSharpCode.SharpZipLib.Tar;
    using TensorFlow;
    using Console = Colorful.Console;

    /// <summary>   A program. </summary>
    class Program
    {
        /// <summary>   The catalog. </summary>
        private static IEnumerable<CatalogItem> _catalog;
        /// <summary>   The current dir. </summary>
        private static string _currentDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        /// <summary>   The input relative. </summary>
        private static string _input_relative = "test_images/input.jpg";
        /// <summary>   The output relative. </summary>
        private static string _output_relative = "test_images/output.jpg";
        /// <summary>   The input. </summary>
        private static string _input = Path.Combine(_currentDir, _input_relative);
        /// <summary>   The output. </summary>
        private static string _output = Path.Combine(_currentDir, _output_relative);
        /// <summary>   Full pathname of the catalog file. </summary>
        private static string _catalogPath;
        /// <summary>   Full pathname of the model file. </summary>
        private static string _modelPath;
        /// <summary>   The minimum score for object highlighting. </summary>
        private static double MIN_SCORE_FOR_OBJECT_HIGHLIGHTING = 0.01;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Main entry-point for this application. </summary>
        ///
        /// <param name="args"> An array of command-line argument strings. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static void Main(string[] args)
        {
            _catalogPath = DownloadDefaultTexts(_currentDir);
			_modelPath = DownloadDefaultModel(_currentDir);
            _catalog = CatalogUtil.ReadCatalogItems(_catalogPath);
            var fileTuples = new List<(string input, string output)>() { (_input, _output) };
            string modelFile = _modelPath;

            using (var graph = new TFGraph())
            {
                graph.Import(new TFBuffer(File.ReadAllBytes(modelFile)));

                using (var session = new TFSession(graph))
                {
                    Console.WriteLine("Detecting objects", Color.Yellow);

                    foreach (var tuple in fileTuples)
                    {
                        var tensor = ImageUtil.CreateTensorFromImageFile(tuple.input, TFDataType.UInt8);
                        var runner = session.GetRunner();

                        runner
                            .AddInput(graph["image_tensor"][0], tensor)
                            .Fetch(
                                graph["detection_boxes"][0],
                                graph["detection_scores"][0],
                                graph["detection_classes"][0],
                                graph["num_detections"][0]);

                        var output = runner.Run();
                        var boxes = (float[,,])output[0].GetValue();
                        var scores = (float[,])output[1].GetValue();
                        var classes = (float[,])output[2].GetValue();

                        Console.WriteLine("Highlighting object...", Color.Green);
                        DrawBoxesOnImage(boxes, scores, classes, tuple.input, tuple.output, MIN_SCORE_FOR_OBJECT_HIGHLIGHTING);
                        Console.WriteLine($"Done. See {_output_relative}. Press any key", Color.Yellow);
                        Console.ReadKey();
                    }
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Downloads the default texts described by dir. </summary>
        ///
        /// <exception cref="ConfigurationErrorsException"> Thrown when a Configuration Errors error
        ///                                                 condition occurs. </exception>
        ///
        /// <param name="dir">  The dir. </param>
        ///
        /// <returns>   A string. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private static string DownloadDefaultTexts(string dir)
        {
            Console.WriteLine("Downloading default label map");

            string defaultTextsUrl = ConfigurationManager.AppSettings["DefaultTextsUrl"] ?? throw new ConfigurationErrorsException("'DefaultTextsUrl' setting is missing in the configuration file");
            var textsFile = Path.Combine(dir, "mscoco_label_map.pbtxt");
            using (var wc = new WebClient())
            {
                wc.DownloadFile(defaultTextsUrl, textsFile);
            }

            return textsFile;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Downloads the default model described by dir. </summary>
        ///
        /// <exception cref="ConfigurationErrorsException"> Thrown when a Configuration Errors error
        ///                                                 condition occurs. </exception>
        ///
        /// <param name="dir">  The dir. </param>
        ///
        /// <returns>   A string. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private static string DownloadDefaultModel(string dir)
        {
            string defaultModelUrl = ConfigurationManager.AppSettings["DefaultModelUrl"] ?? throw new ConfigurationErrorsException("'DefaultModelUrl' setting is missing in the configuration file");

            var modelFile = Path.Combine(dir, "faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/frozen_inference_graph.pb");
            var zipfile = Path.Combine(dir, "faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz");

            if (File.Exists(modelFile))
                return modelFile;

            if (!File.Exists(zipfile))
            {
                Console.WriteLine("Downloading default model");
                using (var wc = new WebClient())
                {
                    wc.DownloadFile(defaultModelUrl, zipfile);
                }
            }

            ExtractToDirectory(zipfile, dir);
            File.Delete(zipfile);

            return modelFile;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Extracts to directory. </summary>
        ///
        /// <param name="file">         The file. </param>
        /// <param name="targetDir">    Target dir. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private static void ExtractToDirectory(string file, string targetDir)
        {
            Console.WriteLine("Extracting");
            Ensure.That(() => !string.IsNullOrWhiteSpace(file));

            using (Stream inStream = File.OpenRead(file))
            {
                using (Stream gzipStream = new GZipInputStream(inStream))
                {
                    TarArchive tarArchive = TarArchive.CreateInputTarArchive(gzipStream);
                    tarArchive?.ExtractContents(targetDir);
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Draw boxes on an image. </summary>
        ///
        /// <param name="boxes">        The boxes. </param>
        /// <param name="scores">       The scores. </param>
        /// <param name="classes">      The classes. </param>
        /// <param name="inputFile">    The input file. </param>
        /// <param name="outputFile">   The output file. </param>
        /// <param name="minScore">     The minimum score. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private static void DrawBoxesOnImage(float[,,] boxes, float[,] scores, float[,] classes, string inputFile, string outputFile, double minScore)
        {
            var x = boxes.GetLength(0);
            var y = boxes.GetLength(1);
            var z = boxes.GetLength(2);

            float ymin = 0, xmin = 0, ymax = 0, xmax = 0;

            using (var editor = new ImageEditor(inputFile, outputFile))
            {
                for (int i = 0; i < x; i++)
                {
                    for (int j = 0; j < y; j++)
                    {
                        if (scores[i, j] < minScore) 
                            continue;

                        for (int k = 0; k < z; k++)
                        {
                            var box = boxes[i, j, k];
                            switch (k)
                            {
                                case 0:
                                    ymin = box;
                                    break;
                                case 1:
                                    xmin = box;
                                    break;
                                case 2:
                                    ymax = box;
                                    break;
                                case 3:
                                    xmax = box;
                                    break;
                            }
                        }

                        int value = Convert.ToInt32(classes[i, j]);
                        CatalogItem catalogItem = _catalog.FirstOrDefault(item => item.Id == value);

                        if (scores[i, j] * 100 > 80.0)
                        {
                            editor.AddBox(xmin, xmax, ymin, ymax,
                                $"{catalogItem.DisplayName} : {(scores[i, j] * 100):0}%", "green");
                        }
                        else if (scores[i, j] > 50 && scores[i, j] <= 79)
                        {
                            editor.AddBox(xmin, xmax, ymin, ymax,
                                $"{catalogItem.DisplayName} : {(scores[i, j] * 100):0}%", "blue");
                        }
                        else if (scores[i, j] > 35 && scores[i, j] <= 49)
                        {
                            editor.AddBox(xmin, xmax, ymin, ymax,
                                $"{catalogItem.DisplayName} : {(scores[i, j] * 100):0}%", "yellow");
                        }
                        else
                        {
                            editor.AddBox(xmin, xmax, ymin, ymax,
                                $"{catalogItem.DisplayName} : {(scores[i, j] * 100):0}%", "red");
                        }
                    }
                }
            }
        }
    }
}

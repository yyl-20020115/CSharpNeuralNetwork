using CNTK;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using ZedGraph;

namespace LSTMTimeSeriesDemo
{
    using System.Globalization;
    using EnsureThat;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   A lstm time series. </summary>
    ///
    /// <seealso cref="T:System.Windows.Forms.Form"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public partial class LSTMTimeSeries : Form
    {
        /// <summary>   The in dim. </summary>
        int inDim = 5;
        /// <summary>   The ou dim. </summary>
        int ouDim = 1;
        /// <summary>   Size of the batch. </summary>
        int batchSize=100;
        /// <summary>   Name of the features. </summary>
        string featuresName = "feature";
        /// <summary>   Name of the labels. </summary>
        string labelsName = "label";

        /// <summary>   Set the data belongs to. </summary>
        Dictionary<string, (float[][] train, float[][] valid, float[][] test)> DataSet;
        /// <summary>   The model line. </summary>
        private LineItem modelLine;
        /// <summary>   The training data line. </summary>
        private LineItem trainingDataLine;
        /// <summary>   The loss data line. </summary>
        private LineItem lossDataLine;
        /// <summary>   The predicted line. </summary>
        private LineItem predictedLine;
        /// <summary>   The test data line. </summary>
        private LineItem testDataLine;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the LSTMTimeSeriesDemo.LSTMTimeSeries class.
        /// </summary>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public LSTMTimeSeries()
        {
            int timeStep = 5;
            InitializeComponent();
            InitiGraphs();

            //load data in to memory
            var xdata = LinearSpace(0, 100.0, 10000).Select(x => (float)x).ToArray();
            DataSet = LoadWaveDataset(Math.Sin, xdata, inDim, timeStep);
        }

        /// <summary>   Initi graphs. </summary>
        private void InitiGraphs()
        {
            ///Fitness simulation chart
            zedGraphControl1.GraphPane.Title.Text = "Model evaluation";
            zedGraphControl1.GraphPane.XAxis.Title.Text = "Samples";
            zedGraphControl1.GraphPane.YAxis.Title.Text = "Observed/Predicted";

            trainingDataLine =
                new LineItem("Data Points", null, null, Color.Red, SymbolType.None, 1)
                {
                    Symbol =
                    {
                        Fill = new Fill(Color.Red),
                        Size = 1
                    }
                };

            modelLine = new LineItem("Data Points", null, null, Color.Blue, SymbolType.None, 1)
            {
                Symbol =
                {
                    Fill = new Fill(Color.Red),
                    Size = 1
                }
            };

            zedGraphControl2.GraphPane.XAxis.Title.Text = "Training Loss";
            zedGraphControl2.GraphPane.XAxis.Title.Text = "Iteration";
            zedGraphControl2.GraphPane.YAxis.Title.Text = "Loss value";

            lossDataLine =
                new LineItem("Loss values", null, null, Color.Red, SymbolType.Circle, 1)
                {
                    Symbol =
                    {
                        Fill = new Fill(Color.Red),
                        Size = 5
                    }
                };

            //Add line to graph
            zedGraphControl1.GraphPane.CurveList?.Add(trainingDataLine);
            zedGraphControl1.GraphPane.CurveList?.Add(modelLine);
            zedGraphControl1.GraphPane.AxisChange(CreateGraphics());

            zedGraphControl2?.GraphPane.CurveList?.Add(lossDataLine);
            zedGraphControl2?.GraphPane.AxisChange(CreateGraphics());

            zedGraphControl3.GraphPane.Title.Text = "Model testing";
            zedGraphControl3.GraphPane.XAxis.Title.Text = "Samples";
            zedGraphControl3.GraphPane.YAxis.Title.Text = "Observed/Predicted";

            testDataLine =
                new LineItem("Actual Data", null, null, Color.Red, SymbolType.None, 1)
                {
                    Symbol =
                    {
                        Fill = new Fill(Color.Red),
                        Size = 1
                    }
                };

            predictedLine =
                new LineItem("Prediction", null, null, Color.Blue, SymbolType.None, 1)
                {
                    Symbol =
                    {
                        Fill = new Fill(Color.Red),
                        Size = 1
                    }
                };

            zedGraphControl3.GraphPane.CurveList?.Add(testDataLine);
            zedGraphControl3.GraphPane.AxisChange(CreateGraphics());
            zedGraphControl3?.GraphPane.CurveList?.Add(predictedLine);
            zedGraphControl3?.GraphPane.AxisChange(CreateGraphics());
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Event handler. Called by Example for load events. </summary>
        ///
        /// <param name="sender">   . </param>
        /// <param name="e">        Event information. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void Example_Load(object sender, EventArgs e)
        {
            LoadTrainingData(DataSet?["features"].train, DataSet?["label"].train);
            PopulateGraphs(DataSet?["label"].train, DataSet?["label"].test);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Populates the graphs. </summary>
        ///
        /// <exception cref="ArgumentException">    Thrown when one or more arguments have unsupported or
        ///                                         illegal values. </exception>
        ///
        /// <param name="train">    The train. </param>
        /// <param name="test">     The test. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void PopulateGraphs(float[][] train, float[][] test)
        {
            if (train == null)
                throw new ArgumentException("TrainNetwork parameter cannot be null");
            if (test == null)
                throw new ArgumentException("test parameter cannot be null");


            for (int i=0; i<train.Length; i++)
              trainingDataLine?.AddPoint(new PointPair(i + 1, train[i][0]));

            for (int i = 0; i < test.Length; i++)
                testDataLine?.AddPoint(new PointPair(i + 1, test[i][0]));

            zedGraphControl1?.RestoreScale(zedGraphControl1.GraphPane);
            zedGraphControl3?.RestoreScale(zedGraphControl3.GraphPane);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Loads training data. </summary>
        ///
        /// <param name="X">    . </param>
        /// <param name="Y">    . </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void LoadTrainingData(float[][] X, float[][] Y)
        {
            //clear the list first
            listView1.Clear();
            listView1.GridLines = true;
            listView1.HideSelection = false;
            if (X == null || Y == null )
                return;
            
            //add features
            listView1.Columns.Add(new ColumnHeader() {Width=20});
            for (int i=0; i < inDim ;i++)
            {
                var col1 = new ColumnHeader
                {
                    Text = $"x{i + 1}",
                    Width = 70
                };
                listView1.Columns.Add(col1);
            }
            
            //Add label
            var col = new ColumnHeader
            {
                Text = $"y",
                Width = 70
            };
            listView1.Columns.Add(col);

            for (int i = 0; i < 100; i++)
            {
               var itm =  listView1.Items.Add($"{(i+1).ToString()}");
                for (int j = 0; j < X[i].Length; j++)
                    itm.SubItems.Add(X[i][j].ToString(CultureInfo.InvariantCulture));
                itm.SubItems.Add(Y[i][0].ToString(CultureInfo.InvariantCulture));
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Creates a batch. </summary>
        ///
        /// <param name="data">     full data. </param>
        /// <param name="start">    . </param>
        /// <param name="count">    . </param>
        ///
        /// <returns>   A new array of float. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal static float[] CreateBatch(float[][] data, int start, int count)
        {
            var lst = new List<float>();
            for (int i = start; i < start + count; i++)
            {
                if (i >= data.Length)
                    break;

                lst.AddRange(data[i]);
            }
            return lst.ToArray();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Iteration method for enumerating data during iteration process of training.
        /// </summary>
        ///
        /// <exception cref="ArgumentException">    Thrown when one or more arguments have unsupported or
        ///                                         illegal values. </exception>
        ///
        /// <param name="X">        . </param>
        /// <param name="Y">        . </param>
        /// <param name="mMSize">   . </param>
        ///
        /// <returns>
        /// An enumerator that allows foreach to be used to process the next data batches in this
        /// collection.
        /// </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private static IEnumerable<(float[] X, float[] Y)> GetNextDataBatch(float[][] X, float[][] Y, int mMSize)
        {
            if (X == null)
                throw new ArgumentException("X parameter cannot be null");
            if (Y == null)
                throw new ArgumentException("Y parameter cannot be null");

            for (int i = 0; i <= X.Length - 1; i += mMSize)
            {
                var size = X.Length - i;
                if (size > 0 && size > mMSize)
                    size = mMSize;

                var x = CreateBatch(X, i, size);
                var y = CreateBatch(Y, i, size);

                yield return (x, y);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Method of generating wave function y=sin(x) </summary>
        ///
        /// <exception cref="ArgumentException">    Thrown when one or more arguments have unsupported or
        ///                                         illegal values. </exception>
        ///
        /// <param name="fun">          . </param>
        /// <param name="x0">           . </param>
        /// <param name="timeSteps">    . </param>
        /// <param name="timeShift">    . </param>
        ///
        /// <returns>   The wave dataset. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Dictionary<string, (float[][] train, float[][] valid, float[][] test)> 
            LoadWaveDataset(Func<double, double> fun, float[] x0, int timeSteps, int timeShift)
        {
            if (x0 == null)
                throw new ArgumentException("x0 parameter cannot be null");

            ////fill data
            float[] xsin = new float[x0.Length];//all data
            for (int l = 0; l < x0.Length; l++)
                xsin[l] = (float)fun(x0[l]);

            //split data on training and testing part
            var a = new float[xsin.Length - timeShift];
            var b = new float[xsin.Length - timeShift];

            for (int l = 0; l < xsin.Length; l++)
            {
                if (l < xsin.Length - timeShift)
                    a[l] = xsin[l];

                if (l >= timeShift)
                    b[l - timeShift] = xsin[l];
            }

            //make arrays of data
            var a1 = new List<float[]>();
            var b1 = new List<float[]>();
            for (int i = 0; i < a.Length - timeSteps + 1; i++)
            {
                //features
                var row = new float[timeSteps];
                for (int j = 0; j < timeSteps; j++)
                    row[j] = a[i + j];
               
                //create features row
                a1.Add(row);
               
                //label row
                b1.Add(new[] { b[i + timeSteps - 1] });
            }

            //split data into TrainNetwork, validation and test data set
            var xxx = SplitDataForTrainingAndTesting(a1.ToArray());
            var yyy = SplitDataForTrainingAndTesting(b1.ToArray());

            var retVal =
                new Dictionary<string, (float[][] train, float[][] valid, float[][] test)>
                {
                    {"features", xxx},
                    {"label", yyy}
                };
            return retVal;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Split data on training validation and testing data sets. </summary>
        ///
        /// <exception cref="ArgumentException">    Thrown when one or more arguments have unsupported or
        ///                                         illegal values. </exception>
        ///
        /// <param name="data">     full data. </param>
        /// <param name="valSize">  (Optional) percentage amount of validation. </param>
        /// <param name="testSize"> (Optional) percentage amount for testing. </param>
        ///
        /// <returns>   A Tuple. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static (float[][] train, float[][] valid, float[][] test) 
            SplitDataForTrainingAndTesting(float[][] data, float valSize = 0.1f, float testSize = 0.1f)
        {
            if (data == null)
                throw new ArgumentException("data parameter cannot be null");

            //calculate
            var posTest = (int)(data.Length * (1 - testSize));
            var posVal = (int)(posTest * (1 - valSize));
            return (
                data.Skip(0).Take(posVal).ToArray(), 
                data.Skip(posVal).Take(posTest - posVal).ToArray(), 
                data.Skip(posTest).ToArray());
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Enumerates linear space in this collection. </summary>
        ///
        /// <param name="start">    . </param>
        /// <param name="stop">     . </param>
        /// <param name="num">      . </param>
        /// <param name="endpoint"> (Optional) </param>
        ///
        /// <returns>
        /// An enumerator that allows foreach to be used to process linear space in this collection.
        /// </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private static IEnumerable<double> LinearSpace(double start, double stop, int num, bool endpoint = true)
        {
            var result = new List<double>();
            if (num <= 0)
            {
                return result;
            }

            if (endpoint)
            {
                if (num == 1)
                {
                    return new List<double>() { start };
                }

                var step = (stop - start) / (num - 1.0d);
                result = GetRange(0, num).Select(v => (v * step) + start).ToList();
            }
            else
            {
                var step = (stop - start) / num;
                result = GetRange(0, num).Select(v => (v * step) + start).ToList();
            }

            return result;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the ranges in this collection. </summary>
        ///
        /// <param name="start">    . </param>
        /// <param name="count">    . </param>
        ///
        /// <returns>
        /// An enumerator that allows foreach to be used to process the ranges in this collection.
        /// </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private static IEnumerable<double> GetRange(double start, int count)
        {
            return Enumerable.Range((int)start, count).Select(v => (double)v);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Starts the training process of LSTM. </summary>
        ///
        /// <param name="sender">   . </param>
        /// <param name="e">        . </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void btnStart_Click(object sender, EventArgs e)
        {
            int iteration = int.Parse(textBox1.Text);
            batchSize = int.Parse(textBox2.Text);

            progressBar1.Maximum = iteration;
            progressBar1.Value = 1;

            inDim = 5;
            ouDim = 1;
            int hiDim = 1;
            int cellDim = inDim;

            Task.Run(() =>
                TrainNetwork(DataSet, hiDim, cellDim, iteration, batchSize, ReportProgress));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Train network. </summary>
        ///
        /// <param name="dataSet">          Set the data belongs to. </param>
        /// <param name="hiDim">            The higher dim. </param>
        /// <param name="cellDim">          The cell dim. </param>
        /// <param name="iteration">        The iteration. </param>
        /// <param name="batchSize">        Size of the batch. </param>
        /// <param name="progressReport">   The progress report. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void TrainNetwork(Dictionary<string, (float[][] train, float[][] valid, float[][] test)> dataSet, 
            int hiDim, int cellDim, int iteration, int batchSize, Action<Trainer, Function, int, DeviceDescriptor> progressReport)
        {
            //split dataset on TrainNetwork, validate and test parts
            var featureSet = dataSet["features"];
            var labelSet = dataSet["label"];

            // build the model
            var feature = Variable.InputVariable(new int[] { inDim }, DataType.Float, featuresName, null, false /*isSparse*/);
            var label = Variable.InputVariable(new int[] { ouDim }, DataType.Float, labelsName, new List<CNTK.Axis>() { CNTK.Axis.DefaultBatchAxis() }, false);
            var lstmModel = LSTMHelper.CreateModel(feature, ouDim, hiDim, cellDim, DeviceDescriptor.CPUDevice, "timeSeriesOutput");
            Function trainingLoss = CNTKLib.SquaredError(lstmModel, label, "squarederrorLoss");
            Function prediction = CNTKLib.SquaredError(lstmModel, label, "squarederrorEval");


            // prepare for training
            TrainingParameterScheduleDouble learningRatePerSample = new TrainingParameterScheduleDouble(0.0005, 1);
            TrainingParameterScheduleDouble momentumTimeConstant = CNTKLib.MomentumAsTimeConstantSchedule(256);

            IList<Learner> parameterLearners = new List<Learner>() 
            {
                Learner.MomentumSGDLearner(lstmModel?.Parameters(), learningRatePerSample, momentumTimeConstant, /*unitGainMomentum = */true)
            };

            //create trainer
            var trainer = Trainer.CreateTrainer(lstmModel, trainingLoss, prediction, parameterLearners);

            // TrainNetwork the model
            for (int i = 1; i <= iteration; i++)
            {
                //get the next mini-batch amount of data
                foreach (var batchData in from miniBatchData in 
                    GetNextDataBatch(featureSet.train, labelSet.train, batchSize) 
                    let xValues = Value.CreateBatch(new NDShape(1, inDim), miniBatchData.X, DeviceDescriptor.CPUDevice) 
                    let yValues = Value.CreateBatch(new NDShape(1, ouDim), miniBatchData.Y, DeviceDescriptor.CPUDevice) 
                    select new Dictionary<Variable, Value>
                {
                    { feature, xValues },
                    { label, yValues }
                })
                {
                    //TrainNetwork mini-batch data
                    trainer?.TrainMinibatch(batchData, DeviceDescriptor.CPUDevice);
                }

                if (InvokeRequired)
                {
                    Invoke(new Action(() => progressReport?.Invoke(trainer, lstmModel.Clone(), i, DeviceDescriptor.CPUDevice)));
                }
                else
                {
                    progressReport?.Invoke(trainer, lstmModel.Clone(), i, DeviceDescriptor.CPUDevice);
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Reports the progress. </summary>
        ///
        /// <exception cref="ArgumentException">    Thrown when one or more arguments have unsupported or
        ///                                         illegal values. </exception>
        ///
        /// <param name="trainer">      The trainer. </param>
        /// <param name="model">        The model. </param>
        /// <param name="iteration">    The iteration. </param>
        /// <param name="device">       The device. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        void ReportProgress(Trainer trainer, Function model, int iteration, DeviceDescriptor device)
        {
            if (trainer == null)
                throw new ArgumentException("trainer parameter cannot be null");
            if (model == null)
                throw new ArgumentException("model parameter cannot be null");

            textBox3.Text = iteration.ToString();
            textBox4.Text = trainer?.PreviousMinibatchLossAverage().ToString(CultureInfo.InvariantCulture);
            progressBar1.Value = iteration;

            GraphProgress(trainer, model, iteration, device);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Graph progress. </summary>
        ///
        /// <exception cref="ArgumentException">    Thrown when one or more arguments have unsupported or
        ///                                         illegal values. </exception>
        ///
        /// <param name="trainer">  The trainer. </param>
        /// <param name="model">    The model. </param>
        /// <param name="i">        Zero-based index of the. </param>
        /// <param name="device">   The device. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void GraphProgress(Trainer trainer, Function model, int i, DeviceDescriptor device)
        {
            if (trainer == null)
                throw new ArgumentException("trainer parameter cannot be null");
            if (model == null)
                throw new ArgumentException("model parameter cannot be null");

            EvaluateCurrentModel(trainer, model, i, device);
            TestCurrentModel(trainer, model, i, device);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Tests current model. </summary>
        ///
        /// <exception cref="ArgumentException">    Thrown when one or more arguments have unsupported or
        ///                                         illegal values. </exception>
        ///
        /// <param name="trainer">  The trainer. </param>
        /// <param name="model">    The model. </param>
        /// <param name="i">        Zero-based index of the. </param>
        /// <param name="device">   The device. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void TestCurrentModel(Trainer trainer, Function model, int i, DeviceDescriptor device)
        {
            if (trainer == null)
                throw new ArgumentException("trainer parameter cannot be null");
            if (model == null)
                throw new ArgumentException("model parameter cannot be null");

            //get the next minibatch amount of data
            int sample = 1;
            predictedLine?.Clear();

            foreach (var miniBatchData in GetNextDataBatch(DataSet["features"].test, DataSet["label"].test, batchSize))
            {
                //get data from dataset
                var xValues = Value.CreateBatch<float>(new NDShape(1, inDim), miniBatchData.X, device);

                //model evaluation
                var fea = model.Arguments[0];
                var lab = model.Output;
                
                //evaluation preparation
                var inputDataMap = new Dictionary<Variable, Value>() { { fea, xValues } };
                var outputDataMap = new Dictionary<Variable, Value>() { { lab, null } };
                model.Evaluate(inputDataMap, outputDataMap, device);
                
                //extract the data
                var oData = outputDataMap[lab].GetDenseData<float>(lab);
               
                //show on graph
                foreach (var y in oData)
                    predictedLine?.AddPoint(new PointPair(sample++, y[0]));
            }
            zedGraphControl3?.RestoreScale(zedGraphControl3.GraphPane);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Evaluate current model. </summary>
        ///
        /// <exception cref="ArgumentException">    Thrown when one or more arguments have unsupported or
        ///                                         illegal values. </exception>
        ///
        /// <param name="trainer">  The trainer. </param>
        /// <param name="model">    The model. </param>
        /// <param name="i">        Zero-based index of the. </param>
        /// <param name="device">   The device. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void EvaluateCurrentModel(Trainer trainer, Function model, int i, DeviceDescriptor device)
        {
            if (trainer == null)
                throw new ArgumentException("trainer parameter cannot be null");
            if (model == null)
                throw new ArgumentException("model parameter cannot be null");

            lossDataLine?.AddPoint(new PointPair(i, trainer.PreviousMinibatchLossAverage()));

            //get the next mini-batch amount of data
            int sample = 1;
            modelLine?.Clear();
            foreach (var miniBatchData in GetNextDataBatch(DataSet["features"].train, DataSet["label"].train, batchSize))
            {
                var xValues = Value.CreateBatch<float>(new NDShape(1, inDim), miniBatchData.X, device);
                var yValues = Value.CreateBatch<float>(new NDShape(1, ouDim), miniBatchData.Y, device);

                //model evaluation
                var fea = model.Arguments[0];

                // build the model
                var lab = model.Output;

                var inputDataMap = new Dictionary<Variable, Value>() { { fea, xValues } };
                var outputDataMap = new Dictionary<Variable, Value>() { { lab, null } };
                model.Evaluate(inputDataMap, outputDataMap, device);

                var oData = outputDataMap[lab]?.GetDenseData<float>(lab);

                foreach (var y in oData)
                    modelLine?.AddPoint(new PointPair(sample++, y[0]));
            }
            zedGraphControl1?.RestoreScale(zedGraphControl1.GraphPane);
            zedGraphControl2?.RestoreScale(zedGraphControl2.GraphPane);
        }
    }
}

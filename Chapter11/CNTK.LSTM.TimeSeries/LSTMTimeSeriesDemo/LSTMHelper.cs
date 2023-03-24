using CNTK;
using System;
using System.Collections.Generic;

namespace LSTMTimeSeriesDemo
{
    /// <summary>   A lstm helper class. </summary>
    public static class LSTMHelper
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Stabilizes. </summary>
        ///
        /// <typeparam name="ElementType">  Type of the element type. </typeparam>
        /// <param name="x">        A Variable to process. </param>
        /// <param name="device">   The device. </param>
        ///
        /// <returns>   A Function. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Function Stabilize<ElementType>(Variable x, DeviceDescriptor device)
        {
            bool isFloatType = typeof(ElementType) == typeof(float);
            var f = Constant.Scalar(isFloatType ? 4.0f : (float) 4.0, device);
            var fInv = Constant.Scalar(f.DataType, 1.0 / 4.0f);
            var beta = CNTKLib.ElementTimes(fInv, CNTKLib.Log(Constant.Scalar(f.DataType, 1.0) +
                    CNTKLib.Exp(CNTKLib.ElementTimes(f, new Parameter(new NDShape(), f.DataType, 0.99537863 /* 1/f*ln (e^f-1) */, device)))));

            return CNTKLib.ElementTimes(beta, x);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Lstmp cell with self stabilization. </summary>
        ///
        /// <typeparam name="ElementType">  Type of the element type. </typeparam>
        /// <param name="input">            The input. </param>
        /// <param name="prevOutput">       The previous output. </param>
        /// <param name="prevCellState">    State of the previous cell. </param>
        /// <param name="device">           The device. </param>
        ///
        /// <returns>   A Tuple&lt;Function,Function&gt; </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Tuple<Function, Function> LSTMPCellWithSelfStabilization<ElementType>(Variable input,
             Variable prevOutput, Variable prevCellState, DeviceDescriptor device)
        {
            int outputDim = prevOutput.Shape[0];
            int cellDim = prevCellState.Shape[0];

            bool isFloatType = typeof(ElementType) == typeof(float);
            DataType dataType = isFloatType ? DataType.Float : DataType.Double;

            Func<int, Parameter> createBiasParam;
            if (isFloatType)
                createBiasParam = (dim) => new Parameter(new int[] { dim }, 0.01f, device, "");
            else
                createBiasParam = (dim) => new Parameter(new int[] { dim }, 0.01, device, "");

            uint seed2 = 1;
            Func<int, Parameter> createProjectionParam = (oDim) => new Parameter(new int[] { oDim, NDShape.InferredDimension },
                    dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed2++), device);

            Func<int, Parameter> createDiagWeightParam = (dim) =>
                new Parameter(new int[] { dim }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed2++), device);

            Function stabilizedPrevOutput = Stabilize<ElementType>(prevOutput, device);
            Function stabilizedPrevCellState = Stabilize<ElementType>(prevCellState, device);

            Func<Variable> projectInput = () =>
                createBiasParam(cellDim) + (createProjectionParam(cellDim) * input);

            // Input gate
            Function it = CNTKLib.Sigmoid((Variable)(projectInput() + (createProjectionParam(cellDim) * stabilizedPrevOutput)) +
                    CNTKLib.ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
            Function bit = CNTKLib.ElementTimes(it, CNTKLib.Tanh(projectInput() + (createProjectionParam(cellDim) 
                   * stabilizedPrevOutput)));

            // Forget-me-not gate
            Function ft = CNTKLib.Sigmoid((Variable)(projectInput() + (createProjectionParam(cellDim) * stabilizedPrevOutput)) +
                        CNTKLib.ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
            Function bft = CNTKLib.ElementTimes(ft, prevCellState);

            Function ct = (Variable)bft + bit;

            // Output gate
            Function ot = CNTKLib.Sigmoid((Variable)(projectInput() + (createProjectionParam(cellDim) * stabilizedPrevOutput)) +
                CNTKLib.ElementTimes(createDiagWeightParam(cellDim), Stabilize<ElementType>(ct, device)));
            Function ht = CNTKLib.ElementTimes(ot, CNTKLib.Tanh(ct));

            return new Tuple<Function, Function>((outputDim != cellDim) ? 
                (createProjectionParam(outputDim) * Stabilize<ElementType>(ht, device)) : ht, ct);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Lstmp component with self stabilization. </summary>
        ///
        /// <typeparam name="ElementType">  Type of the element type. </typeparam>
        /// <param name="input">            The input. </param>
        /// <param name="outputShape">      The output shape. </param>
        /// <param name="cellShape">        The cell shape. </param>
        /// <param name="recurrenceHookH">  The recurrence hook h. </param>
        /// <param name="recurrenceHookC">  The recurrence hook c. </param>
        /// <param name="device">           The device. </param>
        ///
        /// <returns>   A Tuple&lt;Function,Function&gt; </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static Tuple<Function, Function> LSTMPComponentWithSelfStabilization<ElementType>(Variable input,
            NDShape outputShape, NDShape cellShape, Func<Variable, Function> recurrenceHookH,
            Func<Variable, Function> recurrenceHookC, DeviceDescriptor device)
        {
            var dh = Variable.PlaceholderVariable(outputShape, input.DynamicAxes);
            var dc = Variable.PlaceholderVariable(cellShape, input.DynamicAxes);
            var LSTMCell = LSTMPCellWithSelfStabilization<ElementType>(input, dh, dc, device);

            var actualDh = recurrenceHookH(LSTMCell.Item1);
            var actualDc = recurrenceHookC(LSTMCell.Item2);

            // Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
            (LSTMCell.Item1).ReplacePlaceholders(new Dictionary<Variable, Variable> { { dh, actualDh }, { dc, actualDc } });

            return new Tuple<Function, Function>(LSTMCell.Item1, LSTMCell.Item2);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Build a one direction recurrent neural network (RNN) with long-short-term-memory (LSTM) cells.
        /// </summary>
        ///
        /// <param name="input">        The input. </param>
        /// <param name="outDim">       The out dim. </param>
        /// <param name="LSTMDim">      The lstm dim. </param>
        /// <param name="cellDim">      The cell dim. </param>
        /// <param name="device">       The device. </param>
        /// <param name="outputName">   Name of the output. </param>
        ///
        /// <returns>   The new model. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static Function CreateModel(Variable input, int outDim, int LSTMDim, int cellDim, DeviceDescriptor device, string outputName)
        {
            Func<Variable, Function> pastValueRecurrenceHook = (x) => CNTKLib.PastValue(x);

            //creating LSTM cell for each input variables
            Function LSTMFunction = LSTMPComponentWithSelfStabilization<float>(input,  new[] { LSTMDim },
                new[] { cellDim }, pastValueRecurrenceHook, pastValueRecurrenceHook, device)?.Item1;

            //after the LSTM sequence is created return the last cell in order to continue generating the network
            Function lastCell = CNTKLib.SequenceLast(LSTMFunction);

            //implement drop out for 10%
            var dropOut = CNTKLib.Dropout(lastCell,0.2, 1);

            //create last dense layer before output
            return FullyConnectedLinearLayer(dropOut, outDim, device, outputName);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Fully connected linear layer. </summary>
        ///
        /// <param name="input">        The input. </param>
        /// <param name="outputDim">    The output dim. </param>
        /// <param name="device">       The device. </param>
        /// <param name="outputName">   (Optional) Name of the output. </param>
        ///
        /// <returns>   A Function. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static Function FullyConnectedLinearLayer(Variable input, int outputDim, DeviceDescriptor device, string outputName = "")
        {
            System.Diagnostics.Debug.Assert(input.Shape.Rank == 1);
            int inputDim = input.Shape[0];

            var glorotInit = CNTKLib.GlorotUniformInitializer(CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank, CNTKLib.SentinelValueForInferParamInitRank, 1);

            int[] s = { outputDim, inputDim };

            var timesParam = new Parameter(s, DataType.Float, glorotInit, device, "timesParam");
            var timesFunction = CNTKLib.Times(timesParam, input, "times");
            int[] s2 = { outputDim };
            var plusParam = new Parameter(s2, 0.0f, device, "plusParam");
            return CNTKLib.Plus(plusParam, timesFunction, outputName);
        }
    }
}

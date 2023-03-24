/// ------------------------------------------------------
/// SwarmOps - Numeric and heuristic optimization for C#
/// Copyright (C) 2003-2011 Magnus Erik Hvass Pedersen.
/// Please see the file license.txt for license details.
/// SwarmOps on the internet: http://www.Hvass-Labs.org/
/// Portions copyright (C) 2018 Matt R. Cole www.evolvedaisolutions.com
/// ------------------------------------------------------

using System;
using SwarmOps;
using SwarmOps.Optimizers;
using Console = Colorful.Console;
using System.Drawing;


namespace TestCustomProblem
{
    using NodaTime;
    using NodaTime.Extensions;

    /// <summary>
    /// Test an optimizer on a custom problem.
    /// </summary>
    class Program
    {
        // Create an object of the custom problem.
        static Problem Problem = new CustomProblem();

        // Optimization settings.
        static readonly int NumRuns = 50;
        static readonly int DimFactor = 4000;
        static readonly int Dim = Problem.Dimensionality;
        static readonly int NumIterations = DimFactor * Dim;

        // Create optimizer object.
        static Optimizer Optimizer = new DE(Problem);
        //static Optimizer Optimizer = new DESuite(Problem, DECrossover.Variant.Rand1Bin, DESuite.DitherVariant.None);

        // Control parameters for optimizer.
        static readonly double[] Parameters = Optimizer.DefaultParameters;
        //static readonly double[] Parameters = SwarmOps.Optimizers.MOL.Parameters.AllBenchmarks2Dim400IterB;

        // Wrap the optimizer in a logger of result-statistics.
        static readonly bool StatisticsOnlyFeasible = true;
        static Statistics Statistics = new Statistics(Optimizer, StatisticsOnlyFeasible);

        // Wrap it again in a repeater.
        static Repeat Repeat = new RepeatSum(Statistics, NumRuns);

        static void Main(string[] args)
        {
            Console.ReadKey();
            // Initialize PRNG.
            Globals.Random = new RandomOps.MersenneTwister();

            // Set max number of optimization iterations to perform.
            Problem.MaxIterations = NumIterations;

            // Output optimization settings.
            Console.WriteLine("Optimizer: {0}", Optimizer.Name, Color.Yellow);
            Console.WriteLine("Using following parameters:", Color.Yellow);
            Tools.PrintParameters(Optimizer, Parameters);
            Console.WriteLine("Number of optimization runs: {0}", NumRuns, Color.Yellow);
            Console.WriteLine("Problem: {0}", Problem.Name, Color.Yellow);
            Console.WriteLine("\tDimensionality: {0}", Dim, Color.Yellow);
            Console.WriteLine("\tNumIterations per run, max: {0}", NumIterations, Color.Yellow);
            Console.WriteLine();

            // Create a fitness trace for tracing the progress of optimization.
            int NumMeanIntervals = 3000;
            FitnessTrace fitnessTrace = new FitnessTraceMean(NumIterations, NumMeanIntervals);
            FeasibleTrace feasibleTrace = new FeasibleTrace(NumIterations, NumMeanIntervals, fitnessTrace);

            // Assign the fitness trace to the optimizer.
            Optimizer.FitnessTrace = feasibleTrace;

            // Starting-time.
            ZonedDateTime t1 = LocalDateTime.FromDateTime(DateTime.Now).InUtc();

            // Perform optimizations.
            double fitness = Repeat.Fitness(Parameters);

            ZonedDateTime t2 = LocalDateTime.FromDateTime(DateTime.Now).InUtc();
            Duration diff = t2.ToInstant() - t1.ToInstant();

            if (Statistics.FeasibleFraction > 0)
            {
                // Compute result-statistics.
                Statistics.Compute();

                // Output best result, as well as result-statistics.
                Console.WriteLine("Best feasible solution found:", Color.Yellow);
                Tools.PrintParameters(Problem, Statistics.BestParameters);
                Console.WriteLine();
                Console.WriteLine("Result Statistics:", Color.Yellow);
                Console.WriteLine("\tFeasible: \t{0} of solutions found.", Tools.FormatPercent(Statistics.FeasibleFraction), Color.Yellow);
                Console.WriteLine("\tBest Fitness: \t{0}", Tools.FormatNumber(Statistics.FitnessMin), Color.Yellow);
                Console.WriteLine("\tWorst: \t\t{0}", Tools.FormatNumber(Statistics.FitnessMax), Color.Yellow);
                Console.WriteLine("\tMean: \t\t{0}", Tools.FormatNumber(Statistics.FitnessMean), Color.Yellow);
                Console.WriteLine("\tStd.Dev.: \t{0}", Tools.FormatNumber(Statistics.FitnessStdDev), Color.Yellow);
                Console.WriteLine();
                Console.WriteLine("Iterations used per run:", Color.Yellow);
                Console.WriteLine("\tMean: {0}", Tools.FormatNumber(Statistics.IterationsMean), Color.Yellow);
            }
            else
            {
                Console.WriteLine("No feasible solutions found.", Color.Red);
            }

            // Output time-usage.
            Console.WriteLine();
            Console.WriteLine("Time usage: {0}", t2 - t1, Color.Yellow);

            // Output fitness and feasible trace.
            string traceFilename = Problem.Name + ".txt";
            fitnessTrace.WriteToFile("FitnessTrace-" + traceFilename);
            feasibleTrace.WriteToFile("FeasibleTrace-" + traceFilename);
            Console.WriteLine("Finished, press any key");
            Console.ReadKey();
        }
    }
}

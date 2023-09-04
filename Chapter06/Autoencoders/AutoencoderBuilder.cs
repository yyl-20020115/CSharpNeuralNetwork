using System.Collections.Generic;


namespace Autoencoders;

public class AutoencoderBuilder
{
    private const double DEFAULT_RATE_BIAS = 0.001;
    private const double DEFAULT_RATE_WEIGHT = 0.001;
    private const double DEFAULT_MOMENTUM = 0.5;
    private List<RestrictedBoltzmannMachineLayer> layers = new ();
    private AutoencoderLearningRate learnrate = new ();
    private IWeightInitializer weightinitializer = new GaussianWeightInitializer();

    public AutoencoderBuilder()
    {

    }

    public void AddBinaryLayer()
    {
        AddLayer(new RestrictedBoltzmannMachineBinaryLayer(1));
    }
    public void AddBinaryLayer(int PSize)
    {
        AddLayer(new RestrictedBoltzmannMachineBinaryLayer(PSize));
    }

    public void AddGaussianLayer()
    {
        AddLayer(new RestrictedBoltzmannMachineGaussianLayer(1));
    }
    public void AddGaussianLayer(int PSize)
    {
        AddLayer(new RestrictedBoltzmannMachineGaussianLayer(PSize));
    }
    public void SetPreTrainingLearningRateWeights(int PWhich, double PLR)
    {
        learnrate.preLearningRateWeights[PWhich] = PLR;
    }
    public void SetPreTrainingLearningRateBiases(int PWhich, double PLR)
    {
        learnrate.preLearningRateBiases[PWhich] = PLR;
    }
    public void SetPreTrainingMomentumWeights(int PWhich, double PMom)
    {
        learnrate.preMomentumWeights[PWhich] = PMom;
    }
    public void SetPreTrainingMomentumBiases(int PWhich, double PMom)
    {
        learnrate.preMomentumBiases[PWhich] = PMom;
    }
    public void SetFineTuningLearningRateWeights(int PWhich, double PLR)
    {
        learnrate.fineLearningRateWeights[PWhich] = PLR;
    }
    public void SetFineTuningLearningRateBiases(int PWhich, double PLR)
    {
        learnrate.fineLearningRateBiases[PWhich] = PLR;
    }




    private void AddLayer(RestrictedBoltzmannMachineLayer PLayer)
    {
        learnrate.preLearningRateBiases.Add(DEFAULT_RATE_BIAS);
        learnrate.preMomentumBiases.Add(DEFAULT_MOMENTUM);
        learnrate.fineLearningRateBiases.Add(DEFAULT_RATE_BIAS);
        if(layers.Count >= 1)
        {
            learnrate.preLearningRateWeights.Add(DEFAULT_RATE_WEIGHT);
            learnrate.preMomentumWeights.Add(DEFAULT_MOMENTUM);
            learnrate.fineLearningRateWeights.Add(DEFAULT_RATE_WEIGHT);
        }
        layers.Add(PLayer);
    }

    public Autoencoder Build() => new Autoencoder(layers, learnrate, weightinitializer);
}

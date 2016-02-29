(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/Kaggle/KaggleData.m"


SeedRandom[1234];
KaggleNet={
   PadFilter[2],Convolve2DToFilterBankInit[32,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[32,32,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[32,64,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   Adaptor3DTo1D[64,4,4],
   FullyConnected1DTo1DInit[64*4*4,4]
};


wl=KaggleNet;
TrainingHistory={};
ValidationHistory={};
\[Lambda]=.0001;


TrainKaggleNet:=MiniBatchGradientDescent[
      wl,FaceImages[[1;;-501]],FaceLabels[[1;;-501]],
      NNGrad,RegressionLoss1D,
        {MaxEpoch->500000,
         ValidationInputs->FaceImages[[-500;;-1]],
         ValidationTargets->FaceLabels[[-500;;-1]],
         StepMonitor->NNCheckpoint["Kaggle\\KaggleNet"],
         Momentum->0.9,MomentumType->"Nesterov",
         InitialLearningRate->\[Lambda]}];

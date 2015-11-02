(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/KaggleNet/KaggleData.m"


SeedRandom[1234];
KaggleNet={
   PadFilter[2],Convolve2DToFilterBankInit[32,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[32,32,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[32,64,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   Adaptor3DTo1D[64,4,4],
   FullyConnected1DTo1DInit[64*4*4,1],
   Logistic
};


wl=KaggleNet;
TrainingHistory={};
ValidationHistory={};
\[Lambda]=.01;


TrainKaggleNet:=MiniBatchGradientDescent[
      wl,KaggleImages[[1;;-1001]],KaggleLabels[[1;;-1001]],
      NNGrad,CrossEntropyLoss,
        {MaxEpoch->500000,
         ValidationInputs->KaggleImages[[-1000;;-1]],
         ValidationTargets->KaggleLabels[[-1000;;-1]],
         StepMonitor->NNCheckpoint["Kaggle\\KaggleNet"],
         InitialLearningRate->\[Lambda]}];

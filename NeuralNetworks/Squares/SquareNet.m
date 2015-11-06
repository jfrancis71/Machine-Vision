(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


randomSamples=Import["C:\\Users\\julian\\Google Drive\\Personal\\Computer Science\\WebMonitor\\Squares\\Squares.wdx"];


SeedRandom[1234];
SquaresNet={
   PadFilter[2],Convolve2DToFilterBankInit[12,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[12,12,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[12,32,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   Adaptor3DTo1D[32,4,4],
   FullyConnected1DTo1DInit[32*4*4,1],
   Logistic};


wl=SquaresNet;
TrainingHistory={};
ValidationHistory={};
\[Lambda]=.01;


TrainSquares:=MiniBatchGradientDescent[
      wl,randomSamples[[1;;4000,1]],randomSamples[[1;;4000,2;;2]],
      NNGrad,CrossEntropyLoss,
        {MaxEpoch->500000,
         ValidationInputs->randomSamples[[4400;;4600,1]],
         ValidationTargets->randomSamples[[4400;;4600,2]],         
         StepMonitor->NNCheckpoint["Squares\\SquaresRecognition"],
         InitialLearningRate->\[Lambda]}];

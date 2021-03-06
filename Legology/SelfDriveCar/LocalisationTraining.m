(* ::Package:: *)

(*

   Ref: https://code.google.com/p/cuda-convnet/
   Ref: https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-18pct.cfg
   Ref: Alex Krizhevsky

   Not completely faithful implementation.
   He uses RELU and a contrast normalisation layer.
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


RawTrainingImages=Rec1[[2;;-1,1]];
RawTrainingLabels=quantData;


SeedRandom[1234];
samp=RandomSample[Transpose[{RawTrainingImages,RawTrainingLabels}]];


images=samp[[All,1]];
labels=Table[
   ReplacePart[ConstantArray[0.,20],(samp[[s,2]]+1)->1],{s,1,Length[samp]}];


SeedRandom[1234];
LocalisationNet={
   PadFilter[2],Convolve2DToFilterBankInit[32,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[32,32,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[32,64,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   Adaptor3DTo1D[64,4,4],
   FullyConnected1DTo1DInit[64*4*4,20],
   Softmax
};


wl=LocalisationNet;
TrainingHistory={};
ValidationHistory={};
\[Lambda]=.01;


LocalisationNetTrain:=MiniBatchGradientDescent[
      wl,images[[1;;3200]],labels[[1;;3200]],
      NNGrad,CrossEntropyLoss,
        {MaxEpoch->35,
         ValidationInputs->images[[3201;;-1]],
         ValidationTargets->labels[[3201;;-1]],
         StepMonitor->NNCheckpoint["SelfDrive\\Localisationv1"],
         Momentum->.9,MomentumType->"Nesterov",
         InitialLearningRate->\[Lambda]}];

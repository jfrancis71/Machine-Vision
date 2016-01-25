(* ::Package:: *)

(*

   Ref: https://code.google.com/p/cuda-convnet/
   Ref: https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-18pct.cfg
   Ref: Alex Krizhevsky

   Not completely faithful implementation.
   He uses RELU and a contrast normalisation layer.
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


RawTrainingImages=TrainingImages;
RawTrainingLabels=TrainingCommands;


SeedRandom[1234];
samp=RandomSample[Transpose[{RawTrainingImages,RawTrainingLabels}]];


SelfDriveImages=samp[[All,1]];
SelfDriveLabels=Table[
   ReplacePart[ConstantArray[0.,3],(samp[[s,2]]+2)->1],{s,1,Length[samp]}];


SeedRandom[1234];
SelfDriveNet={
   PadFilter[2],Convolve2DToFilterBankInit[32,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[32,32,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[32,64,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   Adaptor3DTo1D[64,4,4],
   FullyConnected1DTo1DInit[64*4*4,3],
   Softmax
};


wl=SelfDriveNet;
TrainingHistory={};
ValidationHistory={};
\[Lambda]=.01;


TrainSelfDriveNet:=MiniBatchGradientDescent[
      wl,SelfDriveImages[[1;;700]],SelfDriveLabels[[1;;700]],
      NNGrad,CrossEntropyLoss,
        {MaxEpoch->500000,
         ValidationInputs->SelfDriveImages[[701;;-1]],
         ValidationTargets->SelfDriveLabels[[701;;-1]],
         StepMonitor->NNCheckpoint["SelfDrive\\SelfDriveNet1"],
         Momentum->.9,MomentumType->"Nesterov",
         InitialLearningRate->\[Lambda]}];

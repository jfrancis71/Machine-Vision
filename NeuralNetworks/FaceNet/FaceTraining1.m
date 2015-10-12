(* ::Package:: *)

(*

   Ref: https://code.google.com/p/cuda-convnet/
   Ref: https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-18pct.cfg
   Ref: Alex Krizhevsky

   Not completely faithful implementation.
   He uses RELU and a contrast normalisation layer.
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/FaceNet/FaceData1.m"


SeedRandom[1234];
FaceNet={
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


wl=FaceNet;
TrainingHistory={};
ValidationHistory={};
\[Lambda]=.01;


TrainFaceNet:=MiniBatchGradientDescent[
      wl,FaceImages[[1;;1800]],FaceLabels[[1;;1800]],
      NNGrad,CrossEntropyLoss,
        {MaxEpoch->500000,
         ValidationInputs->FaceImages[[1801;;2000]],
         ValidationTargets->FaceLabels[[1801;;2000]],
         UpdateFunction->NNCheckpoint["Face\\FaceNet1"],
         InitialLearningRate->\[Lambda]}];

(* ::Package:: *)

(*

   Ref: https://code.google.com/p/cuda-convnet/
   Ref: https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-18pct.cfg
   Ref: Alex Krizhevsky

   Not completely faithful implementation.
   He uses RELU and a contrast normalisation layer.
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


chessDat=Import["C:\\Users\\julian\\ImageDataSetsPublic\\ChessPiece\\ChessPieceImages.wdx"];


RawTrainingImages=Flatten[chessDat[[{1,2,4,5}]],1];


(*RawTrainingLabels=Join[
   ConstantArray[0,Length[chessDat[[1]]]],
   ConstantArray[1,Length[chessDat[[2]]]],
   ConstantArray[2,Length[chessDat[[3]]]],
   ConstantArray[3,Length[chessDat[[4]]]],
   ConstantArray[4,Length[chessDat[[5]]]]
];*)


RawTrainingLabels=Join[
   ConstantArray[0,Length[chessDat[[1]]]],
   ConstantArray[1,Length[chessDat[[2]]]],
   ConstantArray[2,Length[chessDat[[4]]]],
   ConstantArray[3,Length[chessDat[[5]]]]];


SeedRandom[1234];
samp=RandomSample[Transpose[{RawTrainingImages,RawTrainingLabels}]];


ChessPieceImages=samp[[All,1]];
ChessPieceLabels=Table[
   ReplacePart[ConstantArray[0.,4],(samp[[s,2]]+1)->1],{s,1,Length[samp]}];


SeedRandom[1234];
ChessPieceNet={
   PadFilter[2],Convolve2DToFilterBankInit[32,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[32,32,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[32,64,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   Adaptor3DTo1D[64,4,4],
   FullyConnected1DTo1DInit[64*4*4,4],
   Softmax
};


wl=ChessPieceNet;
TrainingHistory={};
ValidationHistory={};
\[Lambda]=.01;


TrainChessPiecesNet:=MiniBatchGradientDescent[
      wl,ChessPieceImages[[1;;1400]],ChessPieceLabels[[1;;1400]],
      NNGrad,CrossEntropyLoss,
        {MaxEpoch->500000,
         ValidationInputs->ChessPieceImages[[1401;;-1]],
         ValidationTargets->ChessPieceLabels[[1401;;-1]],
         StepMonitor->NNCheckpoint["ChessPieces\\ChessPiecesMultiple"],
         Momentum->.9,MomentumType->"Nesterov",
         InitialLearningRate->\[Lambda]}];

(* ::Package:: *)

(*
   Some ideas inspired by:
   Ref: http://dp.readthedocs.org/en/master/facialkeypointstutorial/
      Nicholas Leonard?
*)


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
   FullyConnected1DTo1DInit[64*4*4,32],
   Softmax
};


wl=KaggleNet;
TrainingHistory={};
ValidationHistory={};
\[Lambda]=.01;


FaceLHSX=Table[
ReplacePart[ConstantArray[0,32],Round[FaceLabels[[f,1]]]->1.],{f,1,Length[FaceLabels]}];


FaceLHSX=Table[PDF[NormalDistribution[FaceLabels[[f,1]],.5],p],{f,1,Length[FaceLabels]},{p,1,32}];


TrainKaggleNet:=MiniBatchGradientDescent[
      wl,FaceImages[[1;;-501]],FaceLHSX[[1;;-501]],
      NNGrad,CrossEntropyLoss,
        {MaxEpoch->25,
         ValidationInputs->FaceImages[[-500;;-1]],
         ValidationTargets->FaceLHSX[[-500;;-1]],
         StepMonitor->NNCheckpoint["Kaggle\\FaceLHSXGauss"],
         Momentum->0.9,MomentumType->"Nesterov",
         InitialLearningRate->\[Lambda]}];

(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


generateExample[num_]:=Graphics[{
GrayLevel[Random[]],
Opacity[0.2],Rectangle[{-1,-1},{+1,+1}],
Opacity[1],
GrayLevel[Random[]],
Rotate[Line[{{0,-1},{0,+1}}],Random[]*2*\[Pi],{0,0}],
If[num==2,Rotate[Line[{{0,-1},{0,+1}}],Random[]*2*\[Pi],{0,0}],{}]
}
]


convert[gr_]:=ImageData[ImageResize[ColorConvert[Rasterize[gr],"GrayScale"],{32,32}]]


labels=Table[{RandomInteger[]},{20000}];


data=Map[convert[generateExample[#[[1]]+1]]&,labels];


NNMonitor[]:=(ScreenMonitor[];grOutput={grOutput,Map[Max[Abs[#]]&,gw]})


SeedRandom[1234];
LineNet={
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


wl=LineNet;
TrainingHistory={};
ValidationHistory={};
\[Lambda]=.01;


Train:=MiniBatchGradientDescent[
      wl,data[[1;;19000]],labels[[1;;19000]],
      NNGrad,CrossEntropyLoss,
        {MaxEpoch->500000,
         ValidationInputs->data[[19001;;-1]],
         ValidationTargets->labels[[19001;;-1]],
         StepMonitor->NNMonitor,
Momentum->0.9,MomentumType->"Nesterov",
         InitialLearningRate->\[Lambda]}];

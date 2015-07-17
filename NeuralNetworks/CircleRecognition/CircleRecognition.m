(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


circleTrainingImages=Import["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\CircleRecognition\\CircleTrainingImages.wdx"];
circleTrainingLabels=Import["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\CircleRecognition\\CircleTrainingLabels.wdx"];
CircleNetwork={
   Convolve2DToFilterBank[{Convolve2D[0.,r1-.5],Convolve2D[0.,r2-.5]}],
   FilterBankToFilterBank[0.,r4-.5],
   FilterBankTo2D[0.,r6-.5],
   Adaptor2DTo1D[14],
   FullyConnected1DTo1D[{0},{Table[Random[],{196}]-0.5}/196.]};
CircleInputs=circleTrainingImages;
CircleOutputs=Map[{#}&,circleTrainingLabels];
CircleMonitor:=Dynamic[{ColDispImage/@{
   CircleNetwork[[1,1,1,2]],
   CircleNetwork[[1,1,2,2]],
   wl[[1,1,1,2]],
   wl[[1,1,2,2]],
   gw[[1,1,2]]/Max[Abs[gw[[1,1,2]]]],
   gw[[1,2,2]]/Max[Abs[gw[[1,2,2]]]]
},Max[Abs[gw[[1,1,2]]]],
   Max[Abs[gw[[1,2,2]]]]}]
CircleTrain:=AdaptiveGradientDescent[CircleNetwork,CircleInputs,CircleOutputs,Grad,RegressionLoss1D,{MaxLoop->500000}];


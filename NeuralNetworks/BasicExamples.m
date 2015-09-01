(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


(* Examples *)
sqNetwork={
   FullyConnected1DTo1D[{.2,.3},{{2},{3}}],Tanh,
   FullyConnected1DTo1D[{.6},{{1,7}}],Tanh
};
sqInputs=Transpose[{Table[x,{x,-1,1,0.1}]}];sqInputs//MatrixForm;
sqOutputs=sqInputs^2;sqOutputs//MatrixForm;
sqTrain:=AdaptiveGradientDescent[sqNetwork,sqInputs,sqOutputs,NNGrad,RegressionLoss2D,{MaxEpoch->500000}];


(*See Parallel Distributed Processing Volume 1: Foundations, PDP Research Group, page 332 Figure 4*)
(*Achieves excellent solution quickly*)
XORNetwork={
   FullyConnected1DTo1D[{.2,.3},{{2,.3},{1,Random[]-.5}}],Tanh,
   FullyConnected1DTo1D[{.6},{{1,Random[]-.5}}],Tanh
};
XORInputs={{0,0},{0,1},{1,0},{1,1}};XORInputs//MatrixForm;
XOROutputs=Transpose[{{0,1,1,0}}];XOROutputs//MatrixForm;
XORTrain:=AdaptiveGradientDescent[XORNetwork,XORInputs,XOROutputs,NNGrad,RegressionLoss2D,{MaxEpoch->500000}];


MultInputs=Flatten[Table[{a,b},{a,0,1,.1},{b,0,1,.1}],1];MultInputs//MatrixForm;
MultOutputs=Map[{#[[1]]*#[[2]]}&,MultInputs];MultOutputs//MatrixForm;
MultTrain:=AdaptiveGradientDescent[XORNetwork,MultInputs,MultOutputs,NNGrad,RegressionLoss2D,{MaxEoch->500000}];

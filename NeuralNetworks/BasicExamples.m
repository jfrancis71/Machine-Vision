(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


(* Examples *)

(*Learning the square function*)
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


(*Learning to multiply two inputs*)
MultInputs=Flatten[Table[{a,b},{a,0,1,.1},{b,0,1,.1}],1];MultInputs//MatrixForm;
MultOutputs=Map[{#[[1]]*#[[2]]}&,MultInputs];MultOutputs//MatrixForm;
MultTrain:=AdaptiveGradientDescent[XORNetwork,MultInputs,MultOutputs,NNGrad,RegressionLoss2D,{MaxEoch->500000}];


(*Learning circle function*)
circleNetwork={
   FullyConnected1DTo1DInit[2,4],Tanh,
   FullyConnected1DTo1DInit[4,1],Logistic
};
circleInputs=Flatten[Transpose[{Table[{x,y},{x,-1,1,0.1},{y,-1,+1,0.1}]}],2];circleInputs//MatrixForm;
circleOutputs=1.*Map[{Boole[#[[1]]+#[[2]]>1]}&,circleInputs^2];circleOutputs//MatrixForm;
circleTrain:=GradientDescent[circleNetwork,circleInputs,circleOutputs,NNGrad,CrossEntropyLoss,{MaxEpoch->500000}];

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
sqTrain:=NNGradientDescent[sqNetwork,sqInputs,sqOutputs,NNGrad,RegressionLoss2D,{MaxEpoch->500000}];


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


(*
   We fail to learn any useful solution using either of these two networks.
*)
parityInputs=Tuples[{0,1},8];
parityOutputs=Map[{Boole[EvenQ[Total[#]]]}&,parityInputs];
parityNetwork={
   FullyConnected1DTo1DInit[8,256],
   Logistic,
   FullyConnected1DTo1DInit[256,1],
   Logistic
};
parityNetwork={
   FullyConnected1DTo1DInit[8,8],Tanh,
   FullyConnected1DTo1DInit[8,4],Tanh,
   FullyConnected1DTo1DInit[4,4],Tanh,
   FullyConnected1DTo1DInit[4,2],Tanh,
   FullyConnected1DTo1DInit[2,2],Tanh,
   FullyConnected1DTo1DInit[2,1],Logistic
};
parityNetwork={
   FullyConnected1DTo1DInit[8,8],
   Logistic,
   FullyConnected1DTo1DInit[8,1],
   Logistic
};
solvedParityNet={
(* Idea comes form Parallel Distributed Processing p .334 
   by Rumelhard, Hinton and Williams
   Can manually encode, but can't train
*)
   FullyConnected1DTo1D[Table[5+-i*10,{i,1,8}],Table[10,{i,1,8},{j,1,8}]],
   Logistic,
   FullyConnected1DTo1D[{5},
      {Table[If[EvenQ[i],+10,-10],{i,1,8}]}],
   Logistic
};
parityTrain:=
   NNGradientDescent[parityNetwork,parityInputs,parityOutputs,NNGrad,CrossEntropyLoss,{MaxEpoch->50000000,LearningRate->.5,Momentum->0.9,MomentumType->"Nesterov"}];


(*
   Another PDP example, page 340
   Note, we train using just 200 examples. It sucessfully generalises to the other 56 (of which 3 were symmetric)
*)
symInputs=Tuples[{0,1},8];
symOutputs=Map[
   {Boole[#[[1;;4]]==Reverse[#[[5;;8]]]]}&
   ,symInputs];
symNetwork={
   FullyConnected1DTo1DInit[8,2],
   Logistic,
   FullyConnected1DTo1DInit[2,1],
   Logistic};
SeedRandom[1234];
samp=RandomSample[Transpose[{symInputs,symOutputs}]];
symTrain:=NNGradientDescent[symNetwork,samp[[1;;200,1]],samp[[1;;200,2]],NNGrad,CrossEntropyLoss,{MaxEpoch->50000000,LearningRate->.5,Momentum->0.9,MomentumType->"Nesterov"}];


(*
   Attempt at density modelling
   In 70,000 iterations moderate good approximation to prob density, albeit with some noisy features
*)
circInputs=Join[
   Table[th=Random[];
      {Sin[th*2*\[Pi]],Cos[th*2*\[Pi]]},{100}],
   Table[{Random[]-.5,Random[]-.5}*3,{100}]];
circOutputs=Join[Table[{1},{100}],Table[{0},{100}]];
circNetwork={
   FullyConnected1DTo1DInit[2,20],
   Logistic,
   FullyConnected1DTo1DInit[20,1],
   Logistic};
SeedRandom[1234];
samp=RandomSample[Transpose[{circInputs,circOutputs}]];
circTrain:=NNGradientDescent[circNetwork,samp[[1;;200,1]],samp[[1;;200,2]],NNGrad,CrossEntropyLoss,{MaxEpoch->50000000,LearningRate->.5,Momentum->0.9,MomentumType->"Nesterov"}];
circPlot:=ListPlot3D[
   Table[ForwardPropogate[{{x,y}},wl][[1,1]],{x,-2,+2,.1},{y,-2,+2,.1}]]

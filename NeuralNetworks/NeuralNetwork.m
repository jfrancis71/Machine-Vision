(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"


(*
   network is made up of sequence of layers
   layer is made up of biases for each of the units
   followed by the weight vector for each unit,
   so weight is a matrix where each row is the weight vector
   for one particular unit
*)

ForwardPropogation[inputs_,network_]:=(
   Z[0] = inputs;

   For[layerIndex=1,layerIndex<=Length[network],layerIndex++,
      A[layerIndex]=LayerForwardPropogation[Z[layerIndex-1],network[[layerIndex]]];
      Z[layerIndex]=Tanh[A[layerIndex]];
   ];

   Z[layerIndex-1]
)
(*
   Layer L has U units and preceeding layer has P units
   then network layer looks like (U*1,U*P)

   The linear activation layer looks like T*U where T is the
   number of inputs or training examples

   inputs has shape T*I where I is the number of inputs or possibly just from previous layer
   output is of shape T*O
*)
LayerForwardPropogation[inputs_,FullyConnected1DTo1D[layerBiases_,layerWeights_]]:=(
   Z0 = inputs;

   (*Weight-Activation Consistancy check*)
   Assert[(layerWeights[[1]]//Length)==(Transpose[Z0]//Length)]; (* Incoming weight matrix should match up with number of units from previous layer *)
   (*Weight-Weight Consistancy checl*)
   Assert[(layerBiases//Length)==(layerWeights//Length)]; (*Bias on units should match up with number of units from weight layer *)
   A1=Transpose[layerWeights.Transpose[Z0] + layerBiases]
)

(*
   parameters is shaped by layers where each layer is {bias,weights} where weights is a 2D kernel
*)
LayerForwardPropogation[inputs_,Convolve2D[layerBiases_,layerKernel_]]:=(
   Z0 = inputs;

   A1=Map[ListCorrelate[layerKernel,#]&,inputs]+layerBiases
)

BackPropogation[currentParameters_,inputs_,targets_]:=(

   ForwardPropogation[inputs, currentParameters];
   Assert[Length[currentParameters]<=2];
   networkLayers=Length[currentParameters];

   DeltaZ[networkLayers]=2*(Z[networkLayers]-targets); (*We are implicitly assuming a regression loss function*)
   DeltaA[networkLayers]=DeltaZ[networkLayers]*Sech[A[networkLayers]]^2;

   For[layerIndex=networkLayers-1,layerIndex>0,layerIndex--,
      DeltaZ[layerIndex]=Backprop[currentParameters[[layerIndex+1]],inputs,DeltaZ[layerIndex+1]];
      DeltaA[layerIndex]=DeltaZ[layerIndex]*Sech[A[layerIndex]]^2;
   ];
)

Backprop[FullyConnected1DTo1D[biases_,weights_],inputs_,postLayerDeltaA_]:=Transpose[Transpose[weights].Transpose[postLayerDeltaA]]
Backprop[Convolve2D[biases_,weights_],inputs_,postLayerDeltaA_]:=Transpose[Transpose[weights].Transpose[postLayerDeltaA]]

(*
   The linear activation layer has shape T*U 
   DeltaXX refers to the partial derivative of the loss function wrt that neurone activation
      so it has shape T*U
   targets has shape T*O where O is the number of output units
*)
Grad[currentParameters_,inputs_,targets_]:=(
   BackPropogation[currentParameters,inputs,targets];

   Table[
      LayerGrad[currentParameters[[layerIndex]],layerIndex]
      ,{layerIndex,1,Length[currentParameters]}]
)

LayerGrad[FullyConnected1DTo1D[biases_,weights_],layerIndex_]:={Total[Transpose[DeltaA[layerIndex]],{2}],Transpose[DeltaA[layerIndex]].Z[layerIndex-1]}
LayerGrad[Convolve2D[biases_,weights_],layerIndex_]:={Total[DeltaA[layerIndex],3],Apply[Plus,MapThread[ListCorrelate,{DeltaA[layerIndex],Z[layerIndex-1]}]]}

(*This is implicitly a regression loss function*)
Loss1D[parameters_,inputs_,targets_]:=Total[(ForwardPropogation[inputs,parameters]-targets)^2,2]
Loss2D[parameters_,inputs_,targets_]:=Total[(ForwardPropogation[inputs,parameters]-targets)^2,3]

WeightDec[networkWeight_,grad_]:=(
   Assert[Length[networkWeight]==Length[grad]];
   Table[Head[networkWeight[[n]]][networkWeight[[n,1]]-grad[[n,1]],networkWeight[[n,2]]-grad[[n,2]]],{n,1,Length[grad]}])

GradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,\[Lambda]_,maxLoop_:2000]:=(
   Print["Iter: ",Dynamic[loop],"Current Loss", Dynamic[lossF[wl,inputs,targets]]];
   For[wl=initialParameters;loop=1,loop<=maxLoop,loop++,PreemptProtect[wl=WeightDec[wl,\[Lambda]*gradientF[wl,inputs,targets]]]];
   wl )

Visualise[parameters_]:=(

   Z0 = Table[0,{layer1[[2,1]]//Length}];
   Print[Z0//Length," Inputs"];

   layer1=parameters[[1]];
   Assert[(layer1[[2,1]]//Length)==(Z0//Length)]; (* Incoming weight matrix should match up with number of units from previous layer *)
   Assert[(layer1[[1]]//Length)==(layer1[[2]]//Length)]; (*Bias on units should match up with number of units from weight layer *)

   Z1 = Table[0,{layer1[[1]]//Length}];
   Print[Z1//Length," H1 Units"];

   layer2=parameters[[2]];
   Assert[(layer2[[2,1]]//Length)==(Z1//Length)]; (* Incoming weight matrix should match up with number of units from previous layer *)
   Assert[(layer2[[1]]//Length)==(layer2[[2]]//Length)]; (*Bias on units should match up with number of units from weight layer *)
)


(* Examples *)
sqNetwork={
   FullyConnected1DTo1D[{.2,.3},{{2},{3}}],
   FullyConnected1DTo1D[{.6},{{1,7}}]
};
sqInputs=Transpose[{Table[x,{x,0,1,0.1}]}];sqInputs//MatrixForm;
sqOutputs=sqInputs^2;sqOutputs//MatrixForm;
sqTrained:=GradientDescent[sqNetwork,sqInputs,sqOutputs,Grad,Loss1D,.0001,500000];


XORNetwork={
   FullyConnected1DTo1D[{.2,.3,.7},{{2,.3},{3,.2},{1,Random[]-.5}}],
   FullyConnected1DTo1D[{.6},{{1,Random[]-.5,Random[]-.5}}]
};
XORInputs={{0,0},{0,1},{1,0},{1,1}};XORInputs//MatrixForm;
XOROutputs=Transpose[{{0,1,1,0}}];XOROutputs//MatrixForm;
XORTrained:=GradientDescent[XORNetwork,XORInputs,XOROutputs,Grad,Loss1D,.0001,500000];


MultInputs=Flatten[Table[{a,b},{a,0,1,.1},{b,0,1,.1}],1];MultInputs//MatrixForm;
MultOutputs=Map[{#[[1]]*#[[2]]}&,MultInputs];MultOutputs//MatrixForm;
MultTrained:=GradientDescent[XORNetwork,MultInputs,MultOutputs,Grad,Loss1D,.0001,5000000];


edgeNetwork={Convolve2D[0,Table[Random[],{3},{3}]]};
edgeInputs={StandardiseImage["C:\\Users\\Julian\\secure\\My Pictures\\me3.png"]};
edgeOutputs=ForwardPropogation[edgeInputs,{Convolve2D[0,sobelY]}];
edgeTrained:=GradientDescent[edgeNetwork,edgeInputs,edgeOutputs,Grad,Loss2D,.000001,500000]

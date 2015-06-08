(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"


(*
   network is made up of sequence of layers
   layer is made up of biases for each of the units
   followed by the weight vector for each unit,
   so weight is a matrix where each row is the weight vector
   for one particular unit

   Layer L has U units and preceeding layer has P units
   then network layer looks like (U*1,U*P)

   The linear activation layer looks like U*T where T is the
   number of inputs or training examples

   inputs has shape I*T where I is the number of inputs
   output is of shape O*T
*)

FullyConnectedForwardPropogation[inputs_,parameters_]:=(
   Z0 = inputs;

   layer1=parameters[[1]];
   Assert[(layer1[[2,1]]//Length)==(Z0//Length)]; (* Incoming weight matrix should match up with number of units from previous layer *)
   Assert[(layer1[[1]]//Length)==(layer1[[2]]//Length)]; (*Bias on units should match up with number of units from weight layer *)
   A1=layer1[[2]].Z0 + layer1[[1]];
   Z1 = Tanh[A1];

   layer2=parameters[[2]];
   Assert[(layer2[[2,1]]//Length)==(Z1//Length)]; (* Incoming weight matrix should match up with number of units from previous layer *)
   Assert[(layer2[[1]]//Length)==(layer2[[2]]//Length)]; (*Bias on units should match up with number of units from weight layer *)
   A2=layer2[[2]].Z1 + layer2[[1]];
   Z2 = Tanh[A2]
)

(* INCOMPLETE *)
(*
   The linear activation layer has shape U*T 
   DeltaXX refers to the partial derivative of the loss function wrt that neurone activation
      so it has shape U*T
   targets has shape O*T where O is the number of output units
*)
FullyConnectedGrad[currentParameters_,inputs_,targets_]:=(

   FullyConnectedForwardPropogation[inputs, currentParameters];

   DeltaZ2=2*(Z2-targets); (*We are implicitly assuming a regression loss function*)
   DeltaA2=DeltaZ2*Sech[A2]^2;

   DeltaZ1=Transpose[layer2[[2]]].DeltaA2;
   DeltaA1=DeltaZ1*Sech[A1]^2;

   {{Total[DeltaA1,{2}],DeltaA1.Transpose[inputs]},{Total[DeltaA2,{2}],DeltaA2.Transpose[Z1]}}
)

(*This is implicitly a regression loss function*)
FullyConnectedLoss[parameters_,inputs_,targets_]:=Total[(FullyConnectedForwardPropogation[inputs,parameters]-targets)^2,2]
ConvLoss[parameters_,inputs_,targets_]:=Total[(Convolution2DForwardPropogation[inputs,parameters]-targets)^2,3]

GradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,\[Lambda]_,maxLoop_:2000]:=(
   Print["Iter: ",Dynamic[loop],"Current Loss", Dynamic[lossF[wl,inputs,targets]]];
   For[wl=initialParameters;loop=1,loop<=maxLoop,loop++,wl-=\[Lambda]*gradientF[wl,inputs,targets]];
   wl )

(* Note the different format from fullyconnected
   Inputs is shaped by T*Y*X
   parameters is shaped by layers where each layer is {bias,weights} where weights is a 2D kernel
*)
Convolution2DForwardPropogation[inputs_,parameters_]:=
(
   Z0 = inputs;

   layer1=parameters[[1]];
   A1=Map[ListCorrelate[layer1[[2]],#]&,inputs]+layer1[[1]];
   Z1=Tanh[A1]
)

Convolution2DGrad[currentParameters_,inputs_,targets_]:=(
   Convolution2DForwardPropogation[inputs,currentParameters];

   DeltaZ1=2*(Z1-targets); (*We are implicitly assuming a regression loss function*)
   DeltaA1=DeltaZ1*Sech[A1]^2;

   {{Total[DeltaA1,3],Apply[Plus,MapThread[ListCorrelate,{DeltaA1,Z0}]]}}
)

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
   {{.2,.3},{{2},{3}}},
   {{.6},{{1,7}}}
};
sqInputs={Table[x,{x,0,1,0.1}]};sqInputs//MatrixForm;
sqOutputs=sqInputs^2;sqOutputs//MatrixForm;

sqTrained:=GradientDescent[sqNetwork,sqInputs,sqOutputs,FullyConnectedGrad,FullyConnectedLoss,.0001,500000];


XORNetwork={
   {{.2,.3,.7},{{2,.3},{3,.2},{1,Random[]-.5}}},
   {{.6},{{1,Random[]-.5,Random[]-.5}}}
};
XORInputs=Transpose[{{0,0},{0,1},{1,0},{1,1}}];XORInputs//MatrixForm;
XOROutputs={{0,1,1,0}};XOROutputs//MatrixForm;

XORTrained:=GradientDescent[XORNetwork,XORInputs,XOROutputs,FullyConnectedGrad,FullyConnectedLoss,.0001,500000];


MultInputs=Transpose[Flatten[Table[{a,b},{a,0,1,.1},{b,0,1,.1}],1]];MultInputs//MatrixForm;


MultOutputs=Transpose[Map[{#[[1]]*#[[2]]}&,Transpose[MultInputs]]];MultOutputs//MatrixForm;


MultTrained:=GradientDescent[XORNetwork,MultInputs,MultOutputs,FullyConnectedGrad,FullyConnectedLoss,.0001,5000000];


edgeNetwork={{0,Table[Random[],{3},{3}]}};
edgeInputs={StandardiseImage["C:\\Users\\Julian\\secure\\My Pictures\\me3.png"]};


edgeOutputs=Convolution2DForwardPropogation[{edgeInputs},{{0,sobelY}}][[1]];


edgeTrained:=GradientDescent[edgeNetwork,{edgeInputs},{edgeOutputs},Convolution2DGrad,ConvLoss,.000001,500000]

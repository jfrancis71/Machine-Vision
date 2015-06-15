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

BackPropogation[currentParameters_,inputs_,targets_]:=(

   ForwardPropogation[inputs, currentParameters];
   Assert[Length[currentParameters]<=3];
   networkLayers=Length[currentParameters];

   DeltaZ[networkLayers]=2*(Z[networkLayers]-targets); (*We are implicitly assuming a regression loss function*)
   DeltaA[networkLayers]=DeltaZ[networkLayers]*Sech[A[networkLayers]]^2;

   For[layerIndex=networkLayers-1,layerIndex>0,layerIndex--,
      DeltaZ[layerIndex]=Backprop[currentParameters[[layerIndex+1]],DeltaA[layerIndex+1]];
      DeltaA[layerIndex]=DeltaZ[layerIndex]*Sech[A[layerIndex]]^2;
   ];
)

(*
   The linear activation layer has shape T*U 
   DeltaXX refers to the partial derivative of the loss function wrt that neurone activation
      so it has shape T*U
   targets has shape T*O where O is the number of output units
*)
Grad[currentParameters_,inputs_,targets_]:=(

   Assert[Length[inputs]==Length[targets]];

   BackPropogation[currentParameters,inputs,targets];

   Table[
      LayerGrad[currentParameters[[layerIndex]],Z[layerIndex-1],DeltaA[layerIndex]]
      ,{layerIndex,1,Length[currentParameters]}]
)

(*This is implicitly a regression loss function*)
Loss1D[parameters_,inputs_,targets_]:=Total[(ForwardPropogation[inputs,parameters]-targets)^2,2]
Loss2D[parameters_,inputs_,targets_]:=Total[(ForwardPropogation[inputs,parameters]-targets)^2,3]
Loss3D[parameters_,inputs_,targets_]:=Total[(ForwardPropogation[inputs,parameters]-targets)^2,4]

(*
WeightDec[networkWeight_,grad_]:=(
   Assert[Length[networkWeight]==Length[grad]];
   Table[If[Head[networkWeight[[n]]]==List,
      Head[networkWeight[[n]]][networkWeight[[n,1]]-grad[[n,1]],networkWeight[[n,2]]-grad[[n,2]]],
      Head[networkWeight[[n]]][MapThread[WeightDec,{networkWeight[[n]],grad[[n]]}]]],
{n,1,Length[grad]}])*)
WeightDec[networkLayers_List,grad_List]:=MapThread[WeightDec,{networkLayers,grad}]
WeightDec[networkLayer_FullyConnected1DTo1D,grad_]:=FullyConnected1DTo1D[networkLayer[[1]]-grad[[1]],networkLayer[[2]]-grad[[2]]]
WeightDec[networkLayer_Convolve2D,grad_]:=Convolve2D[networkLayer[[1]]-grad[[1]],networkLayer[[2]]-grad[[2]]]
WeightDec[networkLayer_Convolve2DToFilterBank,grad_]:=Convolve2DToFilterBank[WeightDec[networkLayer[[1]],grad]]
WeightDec[networkLayer_FilterBankTo2D,grad_]:=FilterBankTo2D[networkLayer[[1]]-grad[[1]],networkLayer[[2]]-grad[[2]]]
WeightDec[networkLayer_FilterBankToFilterBank,grad_]:=FilterBankToFilterBank[networkLayer[[1]]-grad[[1]],networkLayer[[2]]-grad[[2]]]

GradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,\[Lambda]_,maxLoop_:2000]:=(
   Print["Iter: ",Dynamic[loop],"Current Loss", Dynamic[lossF[wl,inputs,targets]]];
   For[wl=initialParameters;loop=1,loop<=maxLoop,loop++,PreemptProtect[wl=WeightDec[wl,(gw=\[Lambda]*gradientF[wl,inputs,targets])]]];
   wl )

AdaptiveGradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,maxLoop_:2000]:=(
   \[Lambda]=.001;
   Print["Iter: ",Dynamic[loop]," Current Loss ",Dynamic[lossF[wl,inputs,targets]], " \[Lambda]=",Dynamic[\[Lambda]]];
   For[wl=initialParameters;loop=1,loop<=maxLoop,loop++,PreemptProtect[
   twl=WeightDec[wl,gw=\[Lambda] gradientF[wl,inputs,targets]];
   If[lossF[twl,inputs,targets]<lossF[wl,inputs,targets],(wl=twl;\[Lambda]=\[Lambda]*2),(\[Lambda]=\[Lambda]*0.5)];
]])

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


(*Layer Types*)

(*FullyConnected1D Layer*)
(*
   Layer L has U units and preceeding layer has P units
   then network layer looks like (U*1,U*P)

   The linear activation layer looks like T*U where T is the
   number of inputs or training examples

   inputs has shape T*I where I is the number of inputs or possibly just from previous layer
   output is of shape T*O
*)
LayerForwardPropogation[inputs_,FullyConnected1DTo1D[layerBiases_,layerWeights_]]:=(

   (*Weight-Activation Consistancy check*)
   Assert[(layerWeights[[1]]//Length)==(Transpose[inputs]//Length)]; (* Incoming weight matrix should match up with number of units from previous layer *)
   (*Weight-Weight Consistancy checl*)
   Assert[(layerBiases//Length)==(layerWeights//Length)]; (*Bias on units should match up with number of units from weight layer *)
   Transpose[layerWeights.Transpose[inputs] + layerBiases]
)
Backprop[FullyConnected1DTo1D[biases_,weights_],postLayerDeltaA_]:=Transpose[Transpose[weights].Transpose[postLayerDeltaA]]
LayerGrad[FullyConnected1DTo1D[biases_,weights_],layerInputs_,layerOutputDelta_]:={Total[Transpose[layerOutputDelta],{2}],Transpose[layerOutputDelta].layerInputs}

(*Convolve2DLayer*)
(*
   layer is {bias,weights} where weights is a 2D kernel
*)
LayerForwardPropogation[inputs_,Convolve2D[layerBias_,layerKernel_]]:=(

   Map[ListCorrelate[layerKernel,#]&,inputs]+layerBias
)
Backprop[Convolve2D[biases_,weights_],postLayerDeltaA_]:=Transpose[Transpose[weights].Transpose[postLayerDeltaA]]
LayerGrad[Convolve2D[biases_,weights_],layerInputs_,layerOutputDelta_]:={Total[layerOutputDelta,3],Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta,layerInputs}]]}

(*Convolve2DToFilterBankLayer*)
(*
   layer is {{bias,weights},{bias,weights},...,} where weights is a 2D kernel
   Resulting layer is T*F*Y*X
*)
LayerForwardPropogation[inputs_,Convolve2DToFilterBank[filters_]]:=(

   Transpose[Map[LayerForwardPropogation[inputs,#]&,filters],{2,1,3,4}]
)
Backprop[Convolve2DToFilterBank[filters_],postLayerDeltaA_]:=Sum[Transpose[Transpose[filter[[2]]].Transpose[postLayerDeltaA]],{filter,filters}]
LayerGrad[Convolve2DToFilterBank[filters_],layerInputs_,layerOutputDelta_]:=Table[{Total[layerOutputDelta[[All,filterIndex]],3],Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta[[All,filterIndex]],layerInputs}]]},{filterIndex,1,Length[filters]}]

(*FilterBankTo2DLayer*)
LayerForwardPropogation[inputs_,FilterBankTo2D[bias_,weights_]]:=(
  
   weights.Transpose[inputs,{2,1,3,4}]+bias
)
Backprop[FilterBankTo2D[bias_,weights_],postLayerDeltaA_]:=Transpose[Map[#*postLayerDeltaA&,weights],{2,1,3,4}]
LayerGrad[FilterBankTo2D[bias_,weights_],layerInputs_,layerOutputDelta_]:={Total[layerOutputDelta,3],Table[Total[layerOutputDelta*layerInputs[[All,w]],3],{w,1,Length[weights]}]}

(*FilterBankToFilterBankLayer*)
(*slices is meant to indicate one slice in the layer (ie a 2D structure) *)
(*so FilterBankToFilterBank is comprised of a sequence of FilterBankTo2D structures *)
LayerForwardPropogation[inputs_,FilterBankToFilterBank[biases_,weights_]]:=(
   Transpose[weights.Transpose[inputs]+biases]
)
Backprop[FilterBankToFilterBank[biases_,weights_],postLayerDeltaA_]:=
   Total[Table[postLayerDeltaA[[t,o]]*weights[[o,f]],{t,1,Length[postLayerDeltaA]},{f,1,Length[weights[[1]]]},{o,1,Length[weights]}],{3}]
LayerGrad[FilterBankToFilterBank[biases_,weights_],layerInputs_,layerOutputDelta_]:={
   Table[Total[layerOutputDelta[[All,f]],3],{f,1,Length[layerOutputDelta[[1]]]}],
   Total[Table[Total[layerInputs[[t]][[in]]*layerOutputDelta[[t]][[out]],2],{t,1,Length[layerOutputDelta]},{out,1,Length[layerOutputDelta[[1]]]},{in,1,Length[layerInputs[[1]]]}]]}


(* Examples *)
sqNetwork={
   FullyConnected1DTo1D[{.2,.3},{{2},{3}}],
   FullyConnected1DTo1D[{.6},{{1,7}}]
};
sqInputs=Transpose[{Table[x,{x,0,1,0.1}]}];sqInputs//MatrixForm;
sqOutputs=sqInputs^2;sqOutputs//MatrixForm;
sqTrained:=GradientDescent[sqNetwork,sqInputs,sqOutputs,Grad,Loss1D,.0001,500000];


XORNetwork={
   FullyConnected1DTo1D[{.2,.3},{{2,.3},{1,Random[]-.5}}],
   FullyConnected1DTo1D[{.6},{{1,Random[]-.5}}]
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


edgeFilterBankNetwork={Convolve2DToFilterBank[{Convolve2D[0,Table[Random[],{3},{3}]],Convolve2D[0,Table[Random[],{3},{3}]]}]};
edgeFilterBankOutputs=ForwardPropogation[edgeInputs,{Convolve2DToFilterBank[{Convolve2D[0,sobelY],Convolve2D[0,sobelX]}]}];
edgeFilterBankTrained:=GradientDescent[edgeFilterBankNetwork,edgeInputs,edgeFilterBankOutputs,Grad,Loss3D,.000001,500000]


edgeFilterBankTo2DNetwork={FilterBankTo2D[.3,{.3,.5}]};
edgeFilterBankTo2DInputs=edgeFilterBankOutputs;
edgeFilterBankTo2DOutputs=edgeOutputs;
edgeFilterBankTo2DTrained:=GradientDescent[edgeFilterBankTo2DNetwork,edgeFilterBankTo2DInputs,edgeFilterBankTo2DOutputs,Grad,Loss2D,.000001,500000]


Deep1Network=Join[edgeFilterBankNetwork,edgeFilterBankTo2DNetwork];
Deep1Inputs=edgeInputs;
Deep1Outputs=edgeOutputs;
Deep1Trained:=GradientDescent[Deep1Network,Deep1Inputs,Deep1Outputs,Grad,Loss2D,.000001,500000];
Deep1Monitor:=Dynamic[{wl[[2,2]],{Show[Deep1Network[[1,1,1,2]]//ColDispImage,ImageSize->35],Show[Deep1Network[[1,1,2,2]]//ColDispImage,ImageSize->35]},
{{-gw[[1,1,2]]//MatrixForm,-gw[[1,2,2]]//MatrixForm},-gw[[2,2]]}
}]


Deep2Network=Join[edgeFilterBankNetwork,edgeFilterBankTo2DNetwork];
Deep2Inputs=edgeInputs;
Deep2Outputs=(ForwardPropogation[edgeInputs,{Convolve2D[0,sobelX]}]+ForwardPropogation[edgeInputs,{Convolve2D[0,sobelY]}])/2;
Deep2Trained:=AdaptiveGradientDescent[Deep2Network,Deep2Inputs,Deep2Outputs,Grad,Loss2D,500000];
Deep2Monitor:=Dynamic[{{Show[Deep1Network[[1,1,1,2]]//ColDispImage,ImageSize->35],Show[Deep1Network[[1,1,2,2]]//ColDispImage,ImageSize->35]},wl[[2,2]],{Show[wl[[1,1,1,2]]//ColDispImage,ImageSize->35],Show[wl[[1,1,2,2]]//ColDispImage,ImageSize->35]},
{{-gw[[1,1,2]]//Reverse//MatrixForm,-gw[[1,2,2]]//Reverse//MatrixForm},-gw[[2,2]]}
}]


SemNetwork={
   FullyConnected1DTo1D[
      Table[Random[],{h1,1,6}],
      Table[Random[],{h1,1,6},{i1,1,8}]],
   FullyConnected1DTo1D[
      Table[Random[],{o1,1,6}],
      Table[Random[],{o1,1,6},{h1,1,6}]]
};
SemInputs=Select[Tuples[{0,1},8],Count[#,1]==2&];
SemOutputs=Map[Function[in,Flatten[Map[IntegerDigits[First[#]-1,2,3]&,Position[in,1]]]],SemInputs];
SemTrained:=GradientDescent[SemNetwork,SemInputs,SemOutputs,Grad,Loss1D,.0001,500000];


FTBNetwork={
   Convolve2DToFilterBank[{Convolve2D[0,Table[Random[],{3},{3}]]}],
   FilterBankToFilterBank[{.4,.7},{{.3},{.7}}],
   FilterBankTo2D[.3,{.3,.5}]};
FTBInputs=edgeInputs;
FTBOutputs=(ForwardPropogation[edgeInputs,{Convolve2D[0,sobelY]}])^2;
FTBTrained:=AdaptiveGradientDescent[FTBNetwork,FTBInputs,FTBOutputs,Grad,Loss3D,500000];

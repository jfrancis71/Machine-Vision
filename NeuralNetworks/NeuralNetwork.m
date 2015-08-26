(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"


AbortAssert[bool_,message_]:=
   If[bool==False,
      Print[message];Abort[]];


(*
   network is made up of sequence of layers
   layer is made up of biases for each of the units
   followed by the weight vector for each unit,
   so weight is a matrix where each row is the weight vector
   for one particular unit
*)
ForwardPropogation[inputs_,network_]:=(
   L[0] = inputs;

   Module[{layerIndex=1},
      For[layerIndex=1,layerIndex<=Length[network],layerIndex++,
         L[layerIndex]=LayerForwardPropogation[L[layerIndex-1],network[[layerIndex]]];
      ];
      L[layerIndex-1]]
)

BackPropogation[currentParameters_,inputs_,targets_,lossF_]:=(

   ForwardPropogation[inputs, currentParameters];
   networkLayers=Length[currentParameters];

   AbortAssert[Dimensions[L[networkLayers]]==Dimensions[targets],"BackPropogation::Dimensions of outputs and targets should match"];

   DeltaL[networkLayers]=DeltaLoss[lossF,L[networkLayers],targets];

   For[layerIndex=networkLayers,layerIndex>1,layerIndex--,
(* layerIndex refers to layer being back propogated across
   ie computing delta's for layerIndex-1 given layerIndex *)

      Which[
         MatchQ[currentParameters[[layerIndex]],MaxPoolingFilterBankToFilterBank],
            DeltaL[layerIndex-1]=Backprop[currentParameters[[layerIndex]],L[layerIndex-1],L[layerIndex],DeltaL[layerIndex]];,
         MatchQ[currentParameters[[layerIndex]],Softmax],
            DeltaL[layerIndex-1]=Backprop[currentParameters[[layerIndex]],L[layerIndex],DeltaL[layerIndex]];,
         MatchQ[currentParameters[[layerIndex]],Tanh],
            DeltaL[layerIndex-1]=Backprop[currentParameters[[layerIndex]],L[layerIndex-1],DeltaL[layerIndex]];,
         MatchQ[currentParameters[[layerIndex]],Logistic],
            DeltaL[layerIndex-1]=Backprop[currentParameters[[layerIndex]],L[layerIndex-1],DeltaL[layerIndex]];,
         MatchQ[currentParameters[[layerIndex]],ReLU],
            DeltaL[layerIndex-1]=Backprop[currentParameters[[layerIndex]],L[layerIndex-1],DeltaL[layerIndex]];,
         True,
            DeltaL[layerIndex-1]=Backprop[currentParameters[[layerIndex]],DeltaL[layerIndex]];
      ];

   ];
)

(*
   The linear activation layer has shape T*U 
   DeltaXX refers to the partial derivative of the loss function wrt that neurone activation
      so it has shape T*U
   targets has shape T*O where O is the number of output units
*)
NNGrad[currentParameters_,inputs_,targets_,lossF_]:=(

   AbortAssert[Length[inputs]==Length[targets],"NNGrad::# of Training Labels should equal # of Training Inputs"];

   BackPropogation[currentParameters,inputs,targets,lossF];

   Table[
      LayerGrad[currentParameters[[layerIndex]],L[layerIndex-1],DeltaL[layerIndex]]
      ,{layerIndex,1,Length[currentParameters]}]
)

(*
Deprecated: harder than it looks to implement, be careful with normalising sizes and mixing with only half batches
MemConstrainedGrad[currentParameters_,inputs_,targets_,lossF_]:=
      Total[MapThread[Grad[currentParameters,#1,#2,lossF]&,{Partition[inputs,500,500,{+1,+1},{}],Partition[targets,500,500,{+1,+1},{}]}]]
*)

DeltaLoss[RegressionLoss1D,outputs_,targets_]:=2.0*(outputs-targets)/Length[outputs];
DeltaLoss[RegressionLoss2D,outputs_,targets_]:=2.0*(outputs-targets)/Length[outputs];
DeltaLoss[RegressionLoss3D,outputs_,targets_]:=2.0*(outputs-targets)/Length[outputs];
DeltaLoss[ClassificationLoss,outputs_,targets_]:=-targets*(1.0/outputs)/Length[outputs];

(*This is implicitly a regression loss function*)
RegressionLoss1D[parameters_,inputs_,targets_]:=(outputs=ForwardPropogation[inputs,parameters];AbortAssert[Dimensions[outputs]==Dimensions[targets],"Loss1D::Mismatched Targets and Outputs"];Total[(outputs-targets)^2,2]/Length[inputs]);
RegressionLoss2D[parameters_,inputs_,targets_]:=Total[(ForwardPropogation[inputs,parameters]-targets)^2,3]/Length[inputs];
RegressionLoss3D[parameters_,inputs_,targets_]:=Total[(ForwardPropogation[inputs,parameters]-targets)^2,4]/Length[inputs];
ClassificationLoss[parameters_,inputs_,targets_]:=-Total[Log[Extract[ForwardPropogation[inputs,parameters],Position[targets,1]]]]/Length[inputs];

WeightDec[networkLayers_List,grad_List]:=MapThread[WeightDec,{networkLayers,grad}]
(*
GradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,\[Lambda]_,maxLoop_:2000]:=(
   Print["Iter: ",Dynamic[loop],"Current Loss", Dynamic[loss]];
   For[wl=initialParameters;loop=1,loop<=maxLoop,loop++,PreemptProtect[loss=lossF[wl,inputs,targets];wl=WeightDec[wl,(gw=\[Lambda]*gradientF[wl,inputs,targets,lossF])]]];
   wl );
*)
LineSearch[{\[Lambda]_,v_,current_},objectiveF_]:=
(* This is implicitly a lowest line search *)(
   t\[Lambda]=\[Lambda]*1.1; (*Has an optimism bias*)
   While[(loss=objectiveF[t\[Lambda]*v])>current,t\[Lambda]=t\[Lambda]*.5;AbortAssert[t\[Lambda]>10^-30]];
  {t\[Lambda],loss}
);

GenericGradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,algo_,options_:{}]:=(
   trainingLoss=-\[Infinity];
   {validationInputs,validationTargets,maxLoop,updateF,\[Lambda]} = {ValidationInputs,ValidationTargets,MaxLoop,UpdateFunction,InitialLearningRate} /.
      options /. {ValidationInputs->{},ValidationTargets->{},MaxLoop->20000,UpdateFunction->Identity,InitialLearningRate->.001};
   Print["Iter: ",Dynamic[loop]," Training Loss ",Dynamic[trainingLoss], " \[Lambda]=",Dynamic[\[Lambda]]];
   If[validationInputs!={},Print[" Validation Loss ",Dynamic[validationLoss]]];
   Print[Dynamic[grOutput]];
   For[wl=initialParameters;loop=1,loop<=maxLoop,loop++,
      trainingLoss=lossF[wl,inputs,targets];
      If[validationInputs!={},
         validationLoss=lossF[wl,validationInputs,validationTargets],validationLoss=0.0];
      AppendTo[TrainingHistory,trainingLoss];
      AppendTo[ValidationHistory,validationLoss];
      algo[];
      updateF[];
   ]);

GradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,options_:{}]:=
   GenericGradientDescent[initialParameters,inputs,targets,gradientF,lossF,
      (gw=gradientF[wl,inputs,targets,lossF];
      wl=WeightDec[wl,\[Lambda]*gw];)&,options];

AdaptiveGradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,options_:{}]:=
   GenericGradientDescent[initialParameters,inputs,targets,gradientF,lossF,
      (gw=gradientF[wl,inputs,targets,lossF];
      {\[Lambda],trainingLoss}=LineSearch[{\[Lambda],gw,trainingLoss},lossF[WeightDec[wl,#],inputs,targets]&];
      wl=WeightDec[wl,\[Lambda]*gw];)&,options];

AdaptiveGradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,options_:{}]:=
   GenericGradientDescent[initialParameters,inputs,targets,gradientF,lossF,
      (gw=gradientF[wl,inputs,targets,lossF];
      {\[Lambda],trainingLoss}=LineSearch[{\[Lambda],gw,trainingLoss},Function[g,lossF[WeightDec[wl,g],inputs,targets]]];
      wl=WeightDec[wl,\[Lambda]*gw];)&,options];


MiniBatchGradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,options_:{}]:=
   GenericGradientDescent[initialParameters,inputs,targets,gradientF,lossF,
      (MapThread[
         (
         gw=gradientF[wl,#1,#2,lossF];
         wl=WeightDec[wl,\[Lambda]*gw];)&,
         {Partition[inputs,100],Partition[targets,100]}];)&,options];


Checkpoint[f_,skip_:10]:=Function[{},If[Mod[loop,skip]==1,f[],0]]


(* Note this is quite funky, still needs some modularity thought *)
Persist[filename_]:=Function[{},(
   Export[filename,{TrainingHistory,ValidationHistory,wl,\[Lambda]}];)]


WebMonitor[name_]:=Function[{},(
   Export[StringJoin["C:\\Users\\Julian\\Google Drive\\Personal\\Computer Science\\WebMonitor\\",name,".jpg"],
      Rasterize[{Text[trainingLoss],Text[validationLoss],grOutput=ListPlot[{TrainingHistory,ValidationHistory},PlotRange->All,PlotStyle->{Blue,Green}]},ImageSize->800,RasterSize->1000]];
   Persist[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]][];)]


CheckpointWebMonitor[name_,skip_:10]:=Checkpoint[WebMonitor[name],skip]


(*Assuming a 1 of n target representation*)
ClassificationPerformance[network_,inputs_,targets_]:=
   Module[{proc},
   proc=ForwardPropogation[inputs,network];
   Mean[Boole[Table[Position[proc[[t]],Max[proc[[t]]]]==Position[targets[[t]],Max[targets[[t]]]],{t,1,Length[inputs]}]]]//N
];


(*Layer Types

FullyConnected1DTo1D - Vector of weights required for each neurone in subsequent layer
Convolve2D - Convolves single 2D array to create single 2D array
Convolve2DToFilterBank - Performs several convolutions to create filter bank from single 2D array
FilterBankTo2D - Collapses filter bank to 2D array preserving locality, so single weight for each orginal filter bank
FilterBankToFilterBank - Preserves locality, builds new filter bank. Each new filter requires vector of weights for the previous filter bank
Adaptor2DTo1D - Flattens 2D structure. No weights required. Specify width of orginial 2D structure (so delta signals can be constructed)
ConvolveFilterBankTo2D - Each feature map in the filter bank has its own convolution kernel
ConvolveFilterBankToFilterBank - As ConvolveFilterBankTo2D but applied repeately to build filter bank layer
MaxPoolingFilterBankToFilterBank - 2*2->1*1 downsampling with max applied to filter bank. No parameters
Adaptor3DTo1D - Flattens 3D structure. No weights required. Specify features and width of orginial 3D structure (so delta signals can be constructed)
Softmax - Softmax layer, no weights
Tanh - Simple 1<->1 Non linear layer
ReLU - Rectified Linear Unit layer
Logistic - Logistic Activation layer
*)


(*FullyConnected1D Layer*)
(*
   Layer L has U units and preceeding layer has P units
   then network layer looks like (U*1,U*P)

   The linear activation layer looks like T*U where T is the
   number of inputs or training examples

   inputs has shape T*I where I is the number of inputs or possibly just from previous layer
   output is of shape T*O
*)
SyntaxInformation[FullyConnected1DTo1D]={"ArgumentsPattern"->{_,_}};
LayerForwardPropogation[inputs_,FullyConnected1DTo1D[layerBiases_,layerWeights_]]:=(

   AbortAssert[(layerWeights[[1]]//Length)==(Transpose[inputs]//Length),"FullyConnected1DTo1D::Weight-Activation Error"];
   AbortAssert[(layerBiases//Length)==(layerWeights//Length),"FullyConnected1DTo1D::Weight-Weight Error"];
   Transpose[layerWeights.Transpose[inputs] + layerBiases]
)
Backprop[FullyConnected1DTo1D[biases_,weights_],postLayerDeltaA_]:=postLayerDeltaA.weights
LayerGrad[FullyConnected1DTo1D[biases_,weights_],layerInputs_,layerOutputDelta_]:={Total[Transpose[layerOutputDelta],{2}],Transpose[layerOutputDelta].layerInputs};
WeightDec[networkLayer_FullyConnected1DTo1D,grad_]:=FullyConnected1DTo1D[networkLayer[[1]]-grad[[1]],networkLayer[[2]]-grad[[2]]];

(*Convolve2DLayer*)
(*
   layer is {bias,weights} where weights is a 2D kernel
*)
SyntaxInformation[Convolve2D]={"ArgumentsPattern"->{_,_}};
LayerForwardPropogation[inputs_,Convolve2D[layerBias_,layerKernel_]]:=(
   ListCorrelate[{layerKernel},inputs]+layerBias
);
Backprop[Convolve2D[biases_,weights_],postLayerDeltaA_]:=Table[ListConvolve[weights,postLayerDeltaA[[t]],{+1,-1},0],{t,1,Length[postLayerDeltaA]}]
LayerGrad[Convolve2D[biases_,weights_],layerInputs_,layerOutputDelta_]:={Total[layerOutputDelta,3],Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta,layerInputs}]]};
WeightDec[networkLayer_Convolve2D,grad_]:=Convolve2D[networkLayer[[1]]-grad[[1]],networkLayer[[2]]-grad[[2]]];

(*Convolve2DToFilterBankLayer*)
(*
   layer is {{bias,weights},{bias,weights},...,} where weights is a 2D kernel
   Resulting layer is T*F*Y*X
*)
SyntaxInformation[Convolve2DToFilterBank]={"ArgumentsPattern"->{_}};
LayerForwardPropogation[inputs_,Convolve2DToFilterBank[filters_]]:=(
   Transpose[Map[LayerForwardPropogation[inputs,#]&,filters],{2,1,3,4}]
);
Backprop[Convolve2DToFilterBank[filters_],postLayerDeltaA_]:=Sum[Backprop[filters[[f]],postLayerDeltaA[[All,f]]],{f,1,Length[filters]}]
LayerGrad[Convolve2DToFilterBank[filters_],layerInputs_,layerOutputDelta_]:=Table[{Total[layerOutputDelta[[All,filterIndex]],3],Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta[[All,filterIndex]],layerInputs}]]},{filterIndex,1,Length[filters]}];
WeightDec[networkLayer_Convolve2DToFilterBank,grad_]:=Convolve2DToFilterBank[WeightDec[networkLayer[[1]],grad]];

(*FilterBankTo2DLayer*)
SyntaxInformation[FilterBankTo2D]={"ArgumentsPattern"->{_,_}};
LayerForwardPropogation[inputs_,FilterBankTo2D[bias_,weights_]]:=(
   weights.Transpose[inputs,{2,1,3,4}]+bias
)
Backprop[FilterBankTo2D[bias_,weights_],postLayerDeltaA_]:=Transpose[Map[#*postLayerDeltaA&,weights],{2,1,3,4}]
LayerGrad[FilterBankTo2D[bias_,weights_],layerInputs_,layerOutputDelta_]:={Total[layerOutputDelta,3],
   Table[Total[layerOutputDelta*layerInputs[[All,w]],3],{w,1,Length[weights]}]};
WeightDec[networkLayer_FilterBankTo2D,grad_]:=FilterBankTo2D[networkLayer[[1]]-grad[[1]],networkLayer[[2]]-grad[[2]]];

(*FilterBankToFilterBankLayer*)
(*slices is meant to indicate one slice in the layer (ie a 2D structure) *)
(*so FilterBankToFilterBank is comprised of a sequence of FilterBankTo2D structures *)
SyntaxInformation[FilterBankToFilterBank]={"ArgumentsPattern"->{_,_}};
LayerForwardPropogation[inputs_,FilterBankToFilterBank[biases_,weights_]]:=(
   Transpose[weights.Transpose[inputs]+biases]
)
Backprop[FilterBankToFilterBank[biases_,weights_],postLayerDeltaA_]:=
   Total[Table[postLayerDeltaA[[t,o]]*weights[[o,f]],{t,1,Length[postLayerDeltaA]},{f,1,Length[weights[[1]]]},{o,1,Length[weights]}],{3}]
LayerGrad[FilterBankToFilterBank[biases_,weights_],layerInputs_,layerOutputDelta_]:={
   Table[Total[layerOutputDelta[[All,f]],3],{f,1,Length[layerOutputDelta[[1]]]}],
   Map[Flatten,Transpose[layerOutputDelta,{2,1,3,4}]].Transpose[Map[Flatten,Transpose[layerInputs,{2,1,3,4}]]]};
WeightDec[networkLayer_FilterBankToFilterBank,grad_]:=FilterBankToFilterBank[networkLayer[[1]]-grad[[1]],networkLayer[[2]]-grad[[2]]];

(*Adaptor2DTo1D*)
SyntaxInformation[Adaptor2DTo1D]={"ArgumentsPattern"->{_}};
LayerForwardPropogation[inputs_,Adaptor2DTo1D[width_]]:=(
   Map[Flatten,inputs]
);
Backprop[Adaptor2DTo1D[width_],postLayerDeltaA_]:=
   Map[Partition[#,width]&,postLayerDeltaA];
LayerGrad[Adaptor2DTo1D[width_],layerInputs_,layerOutputDelta_]:=
   Adaptor2DTo1D[width];
WeightDec[networkLayer_Adaptor2DTo1D,grad_]:=Adaptor2DTo1D[networkLayer[[1]]];

(*ConvolveFilterBankTo2D*)
SyntaxInformation[ConvolveFilterBankTo2D]={"ArgumentsPattern"->{_,_}};
LayerForwardPropogation[inputs_,ConvolveFilterBankTo2D[bias_,kernels_]]:=(
   AbortAssert[Length[inputs[[1]]]==Length[kernels],"ConvolveFilterBankTo2D::#Kernels not equal to #Features in input feature map"];
   bias+Sum[ListCorrelate[{kernels[[kernel]]},inputs[[All,kernel]]],
      {kernel,1,Length[kernels]}]);
Backprop[ConvolveFilterBankTo2D[bias_,kernels_],postLayerDeltaA_]:=(
   Transpose[Table[ListConvolve[{kernels[[w]]},postLayerDeltaA,{+1,-1},0],{w,1,Length[kernels]}],{2,1,3,4}]);
LayerGrad[ConvolveFilterBankTo2D[bias_,kernels_],layerInputs_,layerOutputDelta_]:=(
   (*{Total[layerOutputDelta,3],Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta,layerInputs}]]}*)
   {Total[layerOutputDelta,3],Table[Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta,layerInputs[[All,w]]}]],{w,1,Length[kernels]}]})
WeightDec[networkLayer_ConvolveFilterBankTo2D,grad_]:=ConvolveFilterBankTo2D[networkLayer[[1]]-grad[[1]],networkLayer[[2]]-grad[[2]]];

(*ConvolveFilterBankToFilterBank*)
SyntaxInformation[ConvolveFilterBankToFilterBank]={"ArgumentsPattern"->{_}};
LayerForwardPropogation[inputs_,ConvolveFilterBankToFilterBank[filters_]]:=(
   Transpose[Map[LayerForwardPropogation[inputs,#]&,filters],{2,1,3,4}]
);
Backprop[ConvolveFilterBankToFilterBank[filters_],postLayerDeltaA_]:=
   Sum[Backprop[filters[[f]],postLayerDeltaA[[All,f]]],{f,1,Length[filters]}];
LayerGrad[ConvolveFilterBankToFilterBank[filters_],layerInputs_,layerOutputDelta_]:=
   Table[{Total[layerOutputDelta[[All,filterOutputIndex]],3],Table[Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta[[All,filterOutputIndex]],layerInputs[[All,l]]}]],{l,1,Length[layerInputs[[1]]]}]},{filterOutputIndex,1,Length[filters]}]
WeightDec[networkLayer_ConvolveFilterBankToFilterBank,grad_]:=ConvolveFilterBankToFilterBank[WeightDec[networkLayer[[1]],grad]];

(*MaxPoolingFilterBankToFilterBank*)
SyntaxInformation[MaxPoolingFilterBankToFilterBank]={"ArgumentsPattern"->{}};
LayerForwardPropogation[inputs_,MaxPoolingFilterBankToFilterBank]:=
   Map[Function[image,Map[Max,Partition[image,{2,2}],{2}]],inputs,{2}];
UpSample[x_]:=Riffle[temp=Riffle[x,x]//Transpose;temp,temp]//Transpose;
backRouting[previousZ_,nextA_]:=UnitStep[previousZ-Map[UpSample,nextA,{2}]];
Backprop[MaxPoolingFilterBankToFilterBank,layerInputs_,layerOutputs_,postLayerDeltaA_]:=
   backRouting[layerInputs,layerOutputs]*Map[UpSample,postLayerDeltaA,{2}];
LayerGrad[MaxPoolingFilterBankToFilterBank,layerInputs_,layerOutputDelta_]:={};
WeightDec[MaxPoolingFilterBankToFilterBank,grad_]:=MaxPoolingFilterBankToFilterBank;

(*Adaptor3DTo1D*)
(* Helper Function sources from Mathematica on-line documentation regarding example use of Partition *)
unflatten[e_,{d__?((IntegerQ[#]&&Positive[#])&)}]:= 
   Fold[Partition,e,Take[{d},{-1,2,-1}]] /;(Length[e]===Times[d]);
SyntaxInformation[Adaptor3DTo1D]={"ArgumentsPattern"->{_,_,_}};
LayerForwardPropogation[inputs_,Adaptor3DTo1D[features_,width_,height_]]:=(
   Map[Flatten,inputs]
);
Backprop[Adaptor3DTo1D[features_,width_,height_],postLayerDeltaA_]:=
   unflatten[Flatten[postLayerDeltaA],{Length[postLayerDeltaA],features,width,height}];
LayerGrad[Adaptor3DTo1D[features_,width_,height_],layerInputs_,layerOutputDelta_]:=
   Adaptor3DTo1D[features,width,height];
WeightDec[networkLayer_Adaptor3DTo1D,grad_]:=Adaptor3DTo1D[networkLayer[[1]],networkLayer[[2]],networkLayer[[3]]];

(*Softmax*)
SyntaxInformation[Softmax]={"ArgumentsPattern"->{}};
LayerForwardPropogation[inputs_,Softmax]:=Map[Exp[#]/Total[Exp[#]]&,inputs];
Backprop[Softmax,outputs_,postLayerDeltaA_]:=
   Table[
      Sum[postLayerDeltaA[[n,i]]*outputs[[n,i]]*(KroneckerDelta[j,i]-outputs[[n,j]]),{i,1,Length[postLayerDeltaA[[1]]]}],
         {n,1,Length[postLayerDeltaA]},
      {j,1,Length[postLayerDeltaA[[1]]]}];
LayerGrad[Softmax,layerInputs_,layerOutputDelta_]:={};
WeightDec[Softmax,grad_]:=Softmax;

(*Tanh*)
LayerForwardPropogation[inputs_,Tanh]:=Tanh[inputs];
Backprop[Tanh,inputs_,postLayerDeltaA_]:=
   postLayerDeltaA*Sech[inputs]^2;
LayerGrad[Tanh,layerInputs_,layerOutputDelta_]:={};
WeightDec[Tanh,grad_]:=Tanh;

(*ReLU*)
SyntaxInformation[ReLU]={"ArgumentsPattern"->{}};
LayerForwardPropogation[inputs_,ReLU]:=UnitStep[inputs-0]*inputs;
Backprop[ReLU,inputs_,postLayerDeltaA_]:=
   postLayerDeltaA*UnitStep[inputs-0];
LayerGrad[ReLU,layerInputs_,layerOutputDelta_]:={};
WeightDec[ReLU,grad_]:=ReLU;

(*Logistic*)
SyntaxInformation[Logistic]={"ArgumentsPattern"->{}};
LayerForwardPropogation[inputs_,Logistic]:=1./(1.+Exp[-inputs]);
Backprop[Logistic,inputs_,postLayerDeltaA_]:=
   postLayerDeltaA*Exp[inputs]*(1+Exp[inputs])^-2;
LayerGrad[Logistic,layerInputs_,layerOutputDelta_]:={};
WeightDec[Logistic,grad_]:=Logistic;


CheckGrad[lossF_,weight_,inputs_,targets_]:=
   (lossF[WeightDec[wl,-ReplacePart[gw*0.,weight->10^-8]],inputs,targets]-lossF[wl,inputs,targets])/10^-8


(* Some Test Helping Code *)
CheckDeltaSensitivity[levelCheck_:6,cellCheck_:{200,16,3,2},targets_]:={
(* Neuron sensitivity checking code *)
(* Advise save L levels in SaveL before running to prevent interference *)
 (* levelCheck: This is the sensitivity of the output neurones at this level *)
(* So note to check backprop you go one before, eg levelCheck 6 is checking neurons are correct at *)
(* level 6, ie backprop iscorrect for level 7 *)
(* cellCheck: {200,16,3,2} *)
   (10^6)*(ClassificationLoss[wl[[levelCheck+1;;-1]],SaveL[levelCheck]+ReplacePart[(SaveL[levelCheck]*0.),cellCheck->10^-6],targets]-
      ClassificationLoss[wl[[levelCheck+1;;-1]],SaveL[levelCheck],targets]),
   Extract[DeltaL[levelCheck],cellCheck]
}


(* Examples *)
sqNetwork={
   FullyConnected1DTo1D[{.2,.3},{{2},{3}}],Tanh,
   FullyConnected1DTo1D[{.6},{{1,7}}],Tanh
};
sqInputs=Transpose[{Table[x,{x,-1,1,0.1}]}];sqInputs//MatrixForm;
sqOutputs=sqInputs^2;sqOutputs//MatrixForm;
sqTrain:=AdaptiveGradientDescent[sqNetwork,sqInputs,sqOutputs,Grad,RegressionLoss2D,{MaxLoop->500000}];


(*See Parallel Distributed Processing Volume 1: Foundations, PDP Research Group, page 332 Figure 4*)
(*Achieves excellent solution quickly*)
XORNetwork={
   FullyConnected1DTo1D[{.2,.3},{{2,.3},{1,Random[]-.5}}],Tanh,
   FullyConnected1DTo1D[{.6},{{1,Random[]-.5}}],Tanh
};
XORInputs={{0,0},{0,1},{1,0},{1,1}};XORInputs//MatrixForm;
XOROutputs=Transpose[{{0,1,1,0}}];XOROutputs//MatrixForm;
XORTrain:=AdaptiveGradientDescent[XORNetwork,XORInputs,XOROutputs,Grad,RegressionLoss2D,{MaxLoop->500000}];


MultInputs=Flatten[Table[{a,b},{a,0,1,.1},{b,0,1,.1}],1];MultInputs//MatrixForm;
MultOutputs=Map[{#[[1]]*#[[2]]}&,MultInputs];MultOutputs//MatrixForm;
MultTrain:=AdaptiveGradientDescent[XORNetwork,MultInputs,MultOutputs,Grad,RegressionLoss2D,{MaxLoop->500000}];


edgeNetwork={Convolve2D[0,Table[Random[],{3},{3}]],Tanh};
edgeInputs={StandardiseImage["C:\\Users\\Julian\\Google Drive\\Personal\\Pictures\\Dating Photos\\me3.png"]};
edgeOutputs=ForwardPropogation[edgeInputs,{Convolve2D[0,sobelY]}];
edgeTrain:=AdaptiveGradientDescent[edgeNetwork,edgeInputs,edgeOutputs,Grad,RegressionLoss2D,{MaxLoop->500000}];


edgeFilterBankNetwork={Convolve2DToFilterBank[{Convolve2D[0,Table[Random[],{3},{3}]],Tanh,Convolve2D[0,Table[Random[],{3},{3}]]}],Tanh};
edgeFilterBankOutputs=ForwardPropogation[edgeInputs,{Convolve2DToFilterBank[{Convolve2D[0,sobelY],Convolve2D[0,sobelX]}],Tanh}];
edgeFilterBankTrain:=AdaptiveGradientDescent[edgeFilterBankNetwork,edgeInputs,edgeFilterBankOutputs,Grad,RegressionLoss2D,{MaxLoop->500000}];


edgeFilterBankTo2DNetwork={FilterBankTo2D[.3,{.3,.5}],Tanh};
edgeFilterBankTo2DInputs=edgeFilterBankOutputs;
edgeFilterBankTo2DOutputs=edgeOutputs;
edgeFilterBankTo2DTrain:=AdaptiveGradientDescent[edgeFilterBankTo2DNetwork,edgeFilterBankTo2DInputs,edgeFilterBankTo2DOutputs,Grad,RegressionLoss2D,{MaxLoop->500000}];


Deep1Network=Join[edgeFilterBankNetwork,edgeFilterBankTo2DNetwork];
Deep1Inputs=edgeInputs;
Deep1Outputs=edgeOutputs;
Deep1Train:=AdaptiveGradientDescent[Deep1Network,Deep1Inputs,Deep1Outputs,Grad,RegressionLoss2D,{MaxLoop->500000}];
Deep1Monitor:=Dynamic[{wl[[2,2]],{Show[Deep1Network[[1,1,1,2]]//ColDispImage,ImageSize->35],Show[Deep1Network[[1,1,2,2]]//ColDispImage,ImageSize->35]},
{{-gw[[1,1,2]]//MatrixForm,-gw[[1,2,2]]//MatrixForm},-gw[[2,2]]}
}]


Deep2Network=Join[edgeFilterBankNetwork,edgeFilterBankTo2DNetwork];
Deep2Inputs=edgeInputs;
Deep2Outputs=(ForwardPropogation[edgeInputs,{Convolve2D[0,sobelX]}]+ForwardPropogation[edgeInputs,{Convolve2D[0,sobelY]}])/2;
Deep2Train:=AdaptiveGradientDescent[Deep2Network,Deep2Inputs,Deep2Outputs,Grad,RegressionLoss2D,{MaxLoop->500000}];
Deep2Monitor:=Dynamic[{{Show[Deep1Network[[1,1,1,2]]//ColDispImage,ImageSize->35],Show[Deep1Network[[1,1,2,2]]//ColDispImage,ImageSize->35]},wl[[2,2]],{Show[wl[[1,1,1,2]]//ColDispImage,ImageSize->35],Show[wl[[1,1,2,2]]//ColDispImage,ImageSize->35]},
{{-gw[[1,1,2]]//Reverse//MatrixForm,-gw[[1,2,2]]//Reverse//MatrixForm},-gw[[2,2]]}
}]


SemNetwork={
   FullyConnected1DTo1D[
      Table[Random[],{h1,1,6}],
      Table[Random[],{h1,1,6},{i1,1,8}]],Tanh,
   FullyConnected1DTo1D[
      Table[Random[],{o1,1,6}],
      Table[Random[],{o1,1,6},{h1,1,6}]],Tanh
};
SemInputs=Select[Tuples[{0,1},8],Count[#,1]==2&];
SemOutputs=Map[Function[in,Flatten[Map[IntegerDigits[First[#]-1,2,3]&,Position[in,1]]]],SemInputs];
SemTrain:=GradientDescent[SemNetwork,SemInputs,SemOutputs,Grad,RegressionLoss1D,.0001,500000];


SeedRandom[1234];
r1=Partition[RandomReal[{0,1},9],3];r2=Partition[RandomReal[{0,1},9],3];
r3=RandomReal[{0,1},4];
r4=Partition[RandomReal[{0,1},8],2];
r5=Random[];r6=RandomReal[{0,1},4];
FTBNetwork={
   Convolve2DToFilterBank[{Convolve2D[0.,r1-.5],Convolve2D[0.,r2-.5]}],Tanh,
   FilterBankToFilterBank[0.,r4-.5],Tanh
   FilterBankTo2D[0.,r6-.5],Tanh};
FTBInputs=edgeInputs;
FTBOutputs=(ForwardPropogation[edgeInputs,{Convolve2D[0,sobelY]}])^2+(ForwardPropogation[edgeInputs,{Convolve2D[0,sobelX]}])^2;
FTBMonitor:=Dynamic[{ColDispImage/@{
   FTBNetwork[[1,1,1,2]],
   FTBNetwork[[1,1,2,2]],
   wl[[1,1,1,2]],
   wl[[1,1,2,2]],
   gw[[1,1,2]]/Max[Abs[gw[[1,1,2]]]],
   gw[[1,2,2]]/Max[Abs[gw[[1,2,2]]]]
},Max[Abs[gw[[1,1,2]]]],
   Max[Abs[gw[[1,2,2]]]]}]
FTBTrain:=AdaptiveGradientDescent[FTBNetwork,FTBInputs,FTBOutputs,Grad,RegressionLoss2D,{MaxLoss->500000}];


TestNetwork={
   FilterBankToFilterBank[{.4,.7,.1,.7},{{.3,.2},{.7,.3},{.15,.16},{.32,.31}}],Tanh,
   FilterBankTo2D[.3,{.3,.5,.1,.2}],Tanh};
TestInputs=Transpose[{ForwardPropogation[edgeInputs,{Convolve2D[0,sobelY]}],ForwardPropogation[edgeInputs,{Convolve2D[0,sobelX]}]},{2,1,3,4}];
TestOutputs=(ForwardPropogation[edgeInputs,{Convolve2D[0,sobelY]}])^2+(ForwardPropogation[edgeInputs,{Convolve2D[0,sobelX]}])^2;
TestMonitor:=Dynamic[{ColDispImage/@{
   TestNetwork[[1,1,1,2]],
   TestNetwork[[1,1,2,2]],
   wl[[1,1,1,2]],
   wl[[1,1,2,2]],
   gw[[1,1,2]]/Max[Abs[gw[[1,1,2]]]],
   gw[[1,2,2]]/Max[Abs[gw[[1,2,2]]]]
},Max[Abs[gw[[1,1,2]]]],
   Max[Abs[gw[[1,2,2]]]]}]
TestTrain:=AdaptiveGradientDescent[TestNetwork,TestInputs,TestOutputs,Grad,RegressionLoss2D,{MaxLoss->500000}];


SeedRandom[1234];
TestConvolveNetwork={
   Convolve2DToFilterBank[{
      Convolve2D[0,Partition[RandomReal[{0,1},9]-.5,3]],
      Convolve2D[0,Partition[RandomReal[{0,1},9]-.5,3]]}],Tanh,
   ConvolveFilterBankToFilterBank[{
     ConvolveFilterBankTo2D[0,{Partition[RandomReal[{0,1},9]-.5,3],Partition[RandomReal[{0,1},9]-.5,3]}],
     ConvolveFilterBankTo2D[0,{Partition[RandomReal[{0,1},9]-.5,3],Partition[RandomReal[{0,1},9]-.5,3]}],
     ConvolveFilterBankTo2D[0,{Partition[RandomReal[{0,1},9]-.5,3],Partition[RandomReal[{0,1},9]-.5,3]}]
}],Tanh,
   FilterBankTo2D[0.,{.1,.6,.2}],Tanh
};
TestConvolveInputs=edgeInputs/4;
TestConvolveOutputs=edgeInputs[[All,3;;-3,3;;-3]]/4;
TestConvolveTrain:=AdaptiveGradientDescent[TestConvolveNetwork,TestConvolveInputs,TestConvolveOutputs,Grad,RegressionLoss2D,{MaxLoop->500000}];


SeedRandom[1234];
TestMaxNetwork={
   Convolve2DToFilterBank[{
      Convolve2D[0,Partition[RandomReal[{0,1},9]-.5,3]],
      Convolve2D[0,Partition[RandomReal[{0,1},9]-.5,3]]}],Tanh,
   MaxPoolingFilterBankToFilterBank,
   FilterBankTo2D[0,{.1,.5}],Tanh
};
TestMaxInputs=edgeInputs[[All,1;;-2,All]]/4;
TestMaxOutputs=Table[.25*Max[
   edgeInputs[[t,y*2,x*2-1]],
   edgeInputs[[t,y*2,x*2]],
   edgeInputs[[t,y*2-1,x*2-1]],
   edgeInputs[[t,y*2-1,x*2]]]
   ,{t,1,1},{y,1,70},{x,1,63}];
TestMaxTrain:=AdaptiveGradientDescent[TestMaxNetwork,TestMaxInputs,TestMaxOutputs,Grad,RegressionLoss2D,{MaxLoop->500000}];

(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"


AbortAssert[bool_,message_]:=
   If[bool==False,
      Print[message];Abort[]];

LayerName[s_Symbol]:=ToString[SymbolName[s]]
LayerName[h_]:=ToString[Head[h]]


(*
   network is made up of sequence of layers
   layer is made up of biases for each of the units
   followed by the weight vector for each unit,
   so weight is a matrix where each row is the weight vector
   for one particular unit
*)
ForwardPropogateLayers[inputs_,network_]:=
(* We don't include the inputs *)
   Rest[FoldList[
      Timer["ForwardPropogateLayer::"<>LayerName[#2],ForwardPropogateLayer[#1,#2]]&,inputs,network]]

ForwardPropogate[inputs_,network_]:=
   ForwardPropogateLayers[inputs,network][[-1]]


Options[BackPropogation] = { L1A->0.0 };
SyntaxInformation[L1A]={"ArgumentsPattern"->{}};
BackPropogation[currentParameters_,inputs_,targets_,lossF_,OptionsPattern[]]:=(

   xL1A = OptionValue[L1A];

   L = Timer["ForwardPropogateLayers",ForwardPropogateLayers[inputs, currentParameters]];
   networkLayers=Length[currentParameters];

   AbortAssert[Dimensions[L[[networkLayers]]]==Dimensions[targets],"BackPropogation::Dimensions of outputs and targets should match"];

   DeltaL[networkLayers]=DeltaLoss[lossF,L[[networkLayers]],targets];

   For[layerIndex=networkLayers,layerIndex>1,layerIndex--,
(* layerIndex refers to layer being back propogated across
   ie computing delta's for layerIndex-1 given layerIndex *)

      Timer["Backprop Layer "<>LayerName[currentParameters[[layerIndex]]],
      Which[
         MatchQ[currentParameters[[layerIndex]],MaxPoolingFilterBankToFilterBank],
            DeltaL[layerIndex-1]=BackPropogateLayer[currentParameters[[layerIndex]],L[[layerIndex-1]],L[[layerIndex]],DeltaL[layerIndex]];,
         MatchQ[currentParameters[[layerIndex]],Softmax],
            DeltaL[layerIndex-1]=BackPropogateLayer[currentParameters[[layerIndex]],L[[layerIndex]],DeltaL[layerIndex]];,
         MatchQ[currentParameters[[layerIndex]],Tanh],
            DeltaL[layerIndex-1]=BackPropogateLayer[currentParameters[[layerIndex]],L[[layerIndex-1]],DeltaL[layerIndex]];,
         MatchQ[currentParameters[[layerIndex]],Logistic],
            DeltaL[layerIndex-1]=BackPropogateLayer[currentParameters[[layerIndex]],L[[layerIndex-1]],DeltaL[layerIndex]];,
         MatchQ[currentParameters[[layerIndex]],ReLU],
            DeltaL[layerIndex-1]=BackPropogateLayer[currentParameters[[layerIndex]],L[[layerIndex-1]],DeltaL[layerIndex]];,
         MatchQ[currentParameters[[layerIndex]],MaxConvolveFilterBankToFilterBank],
            DeltaL[layerIndex-1]=BackPropogateLayer[currentParameters[[layerIndex]],L[[layerIndex-1]],L[[layerIndex]],DeltaL[layerIndex]];,
         True,
            DeltaL[layerIndex-1]=BackPropogateLayer[currentParameters[[layerIndex]],DeltaL[layerIndex]];
      ];]

      AbortAssert[Dimensions[DeltaL[layerIndex-1]]==Dimensions[L[[layerIndex-1]]]];
      DeltaL[layerIndex-1]+=Sign[L[[layerIndex-1]]]*xL1A;
   ];
)

(*
   The linear activation layer has shape T*U 
   DeltaXX refers to the partial derivative of the loss function wrt that neurone activation
      so it has shape T*U
   targets has shape T*O where O is the number of output units
*)
Options[NNGrad] = {};
NNGrad[currentParameters_,inputs_,targets_,lossF_,opts:OptionsPattern[]]:=(

   AbortAssert[Length[inputs]==Length[targets],"NNGrad::# of Training Labels should equal # of Training Inputs"];

   Timer["BackPropogation Total",
   BackPropogation[currentParameters,inputs,targets,lossF,FilterRules[{opts}, Options[BackPropogation]]];];

   Timer["LayerGad",
   Prepend[
      Table[
         GradLayer[currentParameters[[layerIndex]],L[[layerIndex-1]],DeltaL[layerIndex]]
         ,{layerIndex,2,Length[currentParameters]}],
      GradLayer[currentParameters[[1]],inputs,DeltaL[1]]
   ]]
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
DeltaLoss[CrossEntropyLoss,outputs_,targets_]:=-((-(1-targets)/(1-outputs)) + (targets/outputs))/Length[outputs];

(*This is implicitly a regression loss function*)
RegressionLoss1D[parameters_,inputs_,targets_]:=(outputs=ForwardPropogate[inputs,parameters];AbortAssert[Dimensions[outputs]==Dimensions[targets],"Loss1D::Mismatched Targets and Outputs"];Total[(outputs-targets)^2,2]/Length[inputs]);
RegressionLoss2D[parameters_,inputs_,targets_]:=Total[(ForwardPropogate[inputs,parameters]-targets)^2,3]/Length[inputs];
RegressionLoss3D[parameters_,inputs_,targets_]:=Total[(ForwardPropogate[inputs,parameters]-targets)^2,4]/Length[inputs];
ClassificationLoss[parameters_,inputs_,targets_]:=-Total[Log[Extract[ForwardPropogate[inputs,parameters],Position[targets,1]]]]/Length[inputs];
CrossEntropyLoss[parameters_,inputs_,targets_]:=
   -Total[targets*Log[ForwardPropogate[inputs,parameters]]+(1-targets)*Log[1-ForwardPropogate[inputs,parameters]],2]/Length[inputs];

WeightDec[networkLayers_List,grad_List]:=MapThread[WeightDec,{networkLayers,grad}]

LineSearch[{\[Lambda]_,v_,current_},objectiveF_]:=
(* This is implicitly a lowest line search *)(
   t\[Lambda]=\[Lambda]*1.1; (*Has an optimism bias*)
   While[(loss=objectiveF[t\[Lambda]*v])>current,t\[Lambda]=t\[Lambda]*.5;AbortAssert[t\[Lambda]>10^-30]];
  {t\[Lambda],loss}
);

NullFunction[]=Function[{},(Null)];
Options[GenericGradientDescent] = { MaxEpoch -> 20000,
   StepMonitor->NullFunction, InitialLearningRate->.01,
   ValidationInputs->{},ValidationTargets->{}};
SyntaxInformation[MaxEpoch]={"ArgumentsPattern"->{}};
SyntaxInformation[ValidationInputs]={"ArgumentsPattern"->{}};
SyntaxInformation[ValidationTargets]={"ArgumentsPattern"->{}};
SyntaxInformation[InitialLearningRate]={"ArgumentsPattern"->{}};
GenericGradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,algoF_,opts:OptionsPattern[]]:=(

   trainingLoss=\[Infinity];
   validationLoss=\[Infinity];

   \[Lambda] = OptionValue[InitialLearningRate];
   Print["Epoch: ",Dynamic[epoch]," Training Loss ",Dynamic[trainingLoss], " \[Lambda]=",Dynamic[\[Lambda]]];
   If[OptionValue[ValidationInputs]!={},Print[" Validation Loss ",Dynamic[validationLoss]]];
   Print[Dynamic[grOutput]];
   For[wl=initialParameters;epoch=1,epoch<=OptionValue[MaxEpoch],epoch++,
      algoF[];
      If[OptionValue[ValidationInputs]!={},
        (*Note the validation can be done here as it is assumed it is small enough to compute in memory*)
         validationLoss=lossF[wl,OptionValue[ValidationInputs],OptionValue[ValidationTargets]];
         AppendTo[ValidationHistory,validationLoss];,
         0];
      AppendTo[TrainingHistory,trainingLoss];
      OptionValue[StepMonitor][];
   ]);

GradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,opts:OptionsPattern[]]:=
   GenericGradientDescent[initialParameters,inputs,targets,gradientF,lossF,
      (  gw=gradientF[wl,inputs,targets,lossF,opts];
         wl=WeightDec[wl,\[Lambda]*gw];
         trainingLoss=lossF[wl,inputs,targets];)&,
      opts];

AdaptiveGradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,opts:OptionsPattern[]]:=
   GenericGradientDescent[initialParameters,inputs,targets,gradientF,lossF,
      (  gw=gradientF[wl,inputs,targets,lossF,opts];
         {\[Lambda],trainingLoss}=LineSearch[{\[Lambda],gw,trainingLoss},lossF[WeightDec[wl,#],inputs,targets]&];
         wl=WeightDec[wl,\[Lambda]*gw];
         trainingLoss=lossF[wl,inputs,targets];)&,
      opts];


MiniBatchGradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,opts:OptionsPattern[]]:=(
   Print["Batch #:", Dynamic[batch], " Partial: ",Dynamic[partialTrainingLoss[[-1]]]];
   GenericGradientDescent[initialParameters,inputs,targets,gradientF,lossF,
      (  partialTrainingLoss={};batch=0;
         MapThread[
            (
            batch++;
            gw=gradientF[wl,#1,#2,lossF,opts];
            wl=WeightDec[wl,\[Lambda]*gw];
            AppendTo[partialTrainingLoss,lossF[wl,#1,#2]];)&,
         {Partition[inputs,100],Partition[targets,100]}];
         trainingLoss = Mean[partialTrainingLoss])&,
      opts];)


Checkpoint[f_,skip_:10]:=Function[{},If[Mod[epoch,skip]==1,f[],0]]


NNBaseDir="C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\";
NNBaseDir="C:\\Users\\Julian\\Google Drive\\Personal\\Computer Science\\WebMonitor\\";

(* Note learning rate .01 reference: http://arxiv.org/pdf/1206.5533v2.pdf, page 9 *)
NNInitialise[resourceName_,network_,learningRate_:0.01]:=
   Export[NNBaseDir<>resourceName<>".wdx",{{},{},network,learningRate}]

NNRead[resourceName_String]:=
   (({TrainingHistory,ValidationHistory,wl,\[Lambda]}=
      Import[NNBaseDir<>resourceName<>".wdx"]););

NNWrite[resourceName_String]:=
      Export[NNBaseDir<>resourceName<>".wdx",{TrainingHistory,ValidationHistory,wl,\[Lambda]}];

(* Note Following functions either take no args, or return a function that takes no args
   so they can be used as update functions for example *)
Persist[resourceName_String]:=Function[{},
   NNWrite[resourceName]];

ScreenMonitor[]:=(grOutput=
   ListPlot[
      If[!MatchQ[ValidationHistory,_List],TrainingHistory,{TrainingHistory,ValidationHistory}],
      PlotRange->All,PlotStyle->{Blue,Green}]);

WebMonitor[resourceName_]:=Function[{},
   Export[StringJoin[NNBaseDir,resourceName,".jpg"],
      Rasterize[{Text[trainingLoss],Text[validationLoss],ScreenMonitor[]},ImageSize->800,RasterSize->1000]];];

NNCheckpoint[resourceName_]:=Function[{},(WebMonitor[resourceName][];Persist[resourceName][])];


(*Assuming a 1 of n target representation*)
ClassificationPerformance[network_,inputs_,targets_]:=
   Module[{proc},
   proc=ForwardPropogate[inputs,network];
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
PadFilterBank - Padding for filter banks
RNorm - Local contrast normalisation layer
SubsampleFilterBankToFilterBank - subsamples the filter bank by 2.
PadFilter - Padding for filter
MaxConvolveFilterBankToFilterBank - Each filter in the filter bank is convolved with max operation, neighbourhood 1 (ie 1 on either side)
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
ForwardPropogateLayer[inputs_,FullyConnected1DTo1D[layerBiases_,layerWeights_]]:=(

   AbortAssert[(layerWeights[[1]]//Length)==(Transpose[inputs]//Length),"FullyConnected1DTo1D::Weight-Activation Error"];
   AbortAssert[(layerBiases//Length)==(layerWeights//Length),"FullyConnected1DTo1D::Weight-Weight Error"];
   Transpose[layerWeights.Transpose[inputs] + layerBiases]
)
BackPropogateLayer[FullyConnected1DTo1D[biases_,weights_],postLayerDeltaA_]:=postLayerDeltaA.weights
GradLayer[FullyConnected1DTo1D[biases_,weights_],layerInputs_,layerOutputDelta_]:={Total[Transpose[layerOutputDelta],{2}],Transpose[layerOutputDelta].layerInputs};
WeightDec[networkLayer_FullyConnected1DTo1D,grad_]:=FullyConnected1DTo1D[networkLayer[[1]]-grad[[1]],networkLayer[[2]]-grad[[2]]];
(* For below justification, see http://arxiv.org/pdf/1206.5533v2.pdf page 15 *)
FullyConnected1DTo1DInit[noFromNeurons_,noToNeurones_]:=
   FullyConnected1DTo1D[ConstantArray[0.,noToNeurones],Table[Random[]-.5,{noToNeurones},{noFromNeurons}]/Sqrt[noFromNeurons]]

(*Convolve2DLayer*)
(*
   layer is {bias,weights} where weights is a 2D kernel
*)
SyntaxInformation[Convolve2D]={"ArgumentsPattern"->{_,_}};
ForwardPropogateLayer[inputs_,Convolve2D[layerBias_,layerKernel_]]:=(
   ListCorrelate[{layerKernel},inputs]+layerBias
);
BackPropogateLayer[Convolve2D[biases_,weights_],postLayerDeltaA_]:=Table[ListConvolve[weights,postLayerDeltaA[[t]],{+1,-1},0],{t,1,Length[postLayerDeltaA]}]
GradLayer[Convolve2D[biases_,weights_],layerInputs_,layerOutputDelta_]:={Total[layerOutputDelta,3],Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta,layerInputs}]]};
WeightDec[networkLayer_Convolve2D,grad_]:=Convolve2D[networkLayer[[1]]-grad[[1]],networkLayer[[2]]-grad[[2]]];

(*Convolve2DToFilterBankLayer*)
(*
   layer is {{bias,weights},{bias,weights},...,} where weights is a 2D kernel
   Resulting layer is T*F*Y*X
*)
SyntaxInformation[Convolve2DToFilterBank]={"ArgumentsPattern"->{_}};
ForwardPropogateLayer[inputs_,Convolve2DToFilterBank[filters_]]:=(
   AbortAssert[(inputs[[1]]//Dimensions//Length)==2,"Convolve2DToFilterBank::inputs does not match 2D structure"];
   Transpose[Map[ForwardPropogateLayer[inputs,#]&,filters],{2,1,3,4}]
);
BackPropogateLayer[Convolve2DToFilterBank[filters_],postLayerDeltaA_]:=Sum[BackPropogateLayer[filters[[f]],postLayerDeltaA[[All,f]]],{f,1,Length[filters]}]
GradLayer[Convolve2DToFilterBank[filters_],layerInputs_,layerOutputDelta_]:=Table[{Total[layerOutputDelta[[All,filterIndex]],3],Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta[[All,filterIndex]],layerInputs}]]},{filterIndex,1,Length[filters]}];
WeightDec[networkLayer_Convolve2DToFilterBank,grad_]:=Convolve2DToFilterBank[WeightDec[networkLayer[[1]],grad]];
Convolve2DToFilterBankInit[noNewFilterBank_,filterSize_]:=
   Convolve2DToFilterBank[
      Table[Convolve2D[0.,
            Table[Random[]-.5,{filterSize},{filterSize}]/Sqrt[filterSize*filterSize]],
         {noNewFilterBank}]]

(*FilterBankTo2DLayer*)
SyntaxInformation[FilterBankTo2D]={"ArgumentsPattern"->{_,_}};
ForwardPropogateLayer[inputs_,FilterBankTo2D[bias_,weights_]]:=(
   weights.Transpose[inputs,{2,1,3,4}]+bias
)
BackPropogateLayer[FilterBankTo2D[bias_,weights_],postLayerDeltaA_]:=Transpose[Map[#*postLayerDeltaA&,weights],{2,1,3,4}]
GradLayer[FilterBankTo2D[bias_,weights_],layerInputs_,layerOutputDelta_]:={Total[layerOutputDelta,3],
   Table[Total[layerOutputDelta*layerInputs[[All,w]],3],{w,1,Length[weights]}]};
WeightDec[networkLayer_FilterBankTo2D,grad_]:=FilterBankTo2D[networkLayer[[1]]-grad[[1]],networkLayer[[2]]-grad[[2]]];

(*FilterBankToFilterBankLayer*)
(*slices is meant to indicate one slice in the layer (ie a 2D structure) *)
(*so FilterBankToFilterBank is comprised of a sequence of FilterBankTo2D structures *)
SyntaxInformation[FilterBankToFilterBank]={"ArgumentsPattern"->{_,_}};
ForwardPropogateLayer[inputs_,FilterBankToFilterBank[biases_,weights_]]:=(
   Transpose[weights.Transpose[inputs]+biases]
)
BackPropogateLayer[FilterBankToFilterBank[biases_,weights_],postLayerDeltaA_]:=
   Total[Table[postLayerDeltaA[[t,o]]*weights[[o,f]],{t,1,Length[postLayerDeltaA]},{f,1,Length[weights[[1]]]},{o,1,Length[weights]}],{3}]
GradLayer[FilterBankToFilterBank[biases_,weights_],layerInputs_,layerOutputDelta_]:={
   Table[Total[layerOutputDelta[[All,f]],3],{f,1,Length[layerOutputDelta[[1]]]}],
   Map[Flatten,Transpose[layerOutputDelta,{2,1,3,4}]].Transpose[Map[Flatten,Transpose[layerInputs,{2,1,3,4}]]]};
WeightDec[networkLayer_FilterBankToFilterBank,grad_]:=FilterBankToFilterBank[networkLayer[[1]]-grad[[1]],networkLayer[[2]]-grad[[2]]];

(*Adaptor2DTo1D*)
SyntaxInformation[Adaptor2DTo1D]={"ArgumentsPattern"->{_}};
ForwardPropogateLayer[inputs_,Adaptor2DTo1D[width_]]:=(
   AbortAssert[(inputs[[1,1]]//Length)==width,"Adaptor2DTo1D::widths of inputs does not match Adaptor width"];
   Map[Flatten,inputs]
);
BackPropogateLayer[Adaptor2DTo1D[width_],postLayerDeltaA_]:=
   Map[Partition[#,width]&,postLayerDeltaA];
GradLayer[Adaptor2DTo1D[width_],layerInputs_,layerOutputDelta_]:=
   Adaptor2DTo1D[width];
WeightDec[networkLayer_Adaptor2DTo1D,grad_]:=Adaptor2DTo1D[networkLayer[[1]]];

(*ConvolveFilterBankTo2D*)
SyntaxInformation[ConvolveFilterBankTo2D]={"ArgumentsPattern"->{_,_}};
ForwardPropogateLayer[inputs_,ConvolveFilterBankTo2D[bias_,kernels_]]:=(
   AbortAssert[Length[inputs[[1]]]==Length[kernels],
      "ConvolveFilterBankTo2D::#Kernels ("<>ToString[Length[kernels]]<>") not equal to #Features ("<>ToString[Length[inputs[[1]]]]<>") in input feature map"];
   bias+Sum[ListCorrelate[{kernels[[kernel]]},inputs[[All,kernel]]],
      {kernel,1,Length[kernels]}]);
BackPropogateLayer[ConvolveFilterBankTo2D[bias_,kernels_],postLayerDeltaA_]:=(
   Transpose[Table[ListConvolve[{kernels[[w]]},postLayerDeltaA,{+1,-1},0],{w,1,Length[kernels]}],{2,1,3,4}]);
GradLayer[ConvolveFilterBankTo2D[bias_,kernels_],layerInputs_,layerOutputDelta_]:=(
   (*{Total[layerOutputDelta,3],Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta,layerInputs}]]}*)
   {Total[layerOutputDelta,3],Table[Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta,layerInputs[[All,w]]}]],{w,1,Length[kernels]}]})
WeightDec[networkLayer_ConvolveFilterBankTo2D,grad_]:=ConvolveFilterBankTo2D[networkLayer[[1]]-grad[[1]],networkLayer[[2]]-grad[[2]]];

(*ConvolveFilterBankToFilterBank*)
SyntaxInformation[ConvolveFilterBankToFilterBank]={"ArgumentsPattern"->{_}};
ForwardPropogateLayer[inputs_,ConvolveFilterBankToFilterBank[filters_]]:=(
   Transpose[Map[ForwardPropogateLayer[inputs,#]&,filters],{2,1,3,4}]
);
BackPropogateLayer[ConvolveFilterBankToFilterBank[filters_],postLayerDeltaA_]:=
   Sum[BackPropogateLayer[filters[[f]],postLayerDeltaA[[All,f]]],{f,1,Length[filters]}];
GradLayer[ConvolveFilterBankToFilterBank[filters_],layerInputs_,layerOutputDelta_]:=
   Table[{Total[layerOutputDelta[[All,filterOutputIndex]],3],Table[Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta[[All,filterOutputIndex]],layerInputs[[All,l]]}]],{l,1,Length[layerInputs[[1]]]}]},{filterOutputIndex,1,Length[filters]}]
WeightDec[networkLayer_ConvolveFilterBankToFilterBank,grad_]:=ConvolveFilterBankToFilterBank[WeightDec[networkLayer[[1]],grad]];
ConvolveFilterBankToFilterBankInit[noOldFilterBank_,noNewFilterBank_,filterSize_]:=
   ConvolveFilterBankToFilterBank[
      Table[ConvolveFilterBankTo2D[0.,
            Table[Random[]-.5,{noOldFilterBank},{filterSize},{filterSize}]/Sqrt[noOldFilterBank*filterSize*filterSize]],
         {noNewFilterBank}]]

(*MaxPoolingFilterBankToFilterBank*)
SyntaxInformation[MaxPoolingFilterBankToFilterBank]={"ArgumentsPattern"->{}};
ForwardPropogateLayer[inputs_,MaxPoolingFilterBankToFilterBank]:=
   Map[Function[image,Map[Max,Partition[image,{2,2}],{2}]],inputs,{2}];
UpSample[x_]:=Riffle[temp=Riffle[x,x]//Transpose;temp,temp]//Transpose;
backRouting[previousZ_,nextA_]:=UnitStep[previousZ-Map[UpSample,nextA,{2}]];
BackPropogateLayer[MaxPoolingFilterBankToFilterBank,layerInputs_,layerOutputs_,postLayerDeltaA_]:=
   backRouting[layerInputs,layerOutputs]*Map[UpSample,postLayerDeltaA,{2}];
GradLayer[MaxPoolingFilterBankToFilterBank,layerInputs_,layerOutputDelta_]:={};
WeightDec[MaxPoolingFilterBankToFilterBank,grad_]:=MaxPoolingFilterBankToFilterBank;

(*Adaptor3DTo1D*)
(* Helper Function sources from Mathematica on-line documentation regarding example use of Partition *)
unflatten[e_,{d__?((IntegerQ[#]&&Positive[#])&)}]:= 
   Fold[Partition,e,Take[{d},{-1,2,-1}]] /;(Length[e]===Times[d]);
SyntaxInformation[Adaptor3DTo1D]={"ArgumentsPattern"->{_,_,_}};
ForwardPropogateLayer[inputs_,Adaptor3DTo1D[features_,width_,height_]]:=(
   Map[Flatten,inputs]
);
BackPropogateLayer[Adaptor3DTo1D[features_,width_,height_],postLayerDeltaA_]:=
   unflatten[Flatten[postLayerDeltaA],{Length[postLayerDeltaA],features,width,height}];
GradLayer[Adaptor3DTo1D[features_,width_,height_],layerInputs_,layerOutputDelta_]:=
   Adaptor3DTo1D[features,width,height];
WeightDec[networkLayer_Adaptor3DTo1D,grad_]:=Adaptor3DTo1D[networkLayer[[1]],networkLayer[[2]],networkLayer[[3]]];

(*Softmax*)
SyntaxInformation[Softmax]={"ArgumentsPattern"->{}};
ForwardPropogateLayer[inputs_,Softmax]:=Map[Exp[#]/Total[Exp[#]]&,inputs];
BackPropogateLayer[Softmax,outputs_,postLayerDeltaA_]:=
   Table[
      Sum[postLayerDeltaA[[n,i]]*outputs[[n,i]]*(KroneckerDelta[j,i]-outputs[[n,j]]),{i,1,Length[postLayerDeltaA[[1]]]}],
         {n,1,Length[postLayerDeltaA]},
      {j,1,Length[postLayerDeltaA[[1]]]}];
GradLayer[Softmax,layerInputs_,layerOutputDelta_]:={};
WeightDec[Softmax,grad_]:=Softmax;

(*Tanh*)
ForwardPropogateLayer[inputs_,Tanh]:=Tanh[inputs];
BackPropogateLayer[Tanh,inputs_,postLayerDeltaA_]:=
   postLayerDeltaA*Sech[inputs]^2;
GradLayer[Tanh,layerInputs_,layerOutputDelta_]:={};
WeightDec[Tanh,grad_]:=Tanh;

(*ReLU*)
SyntaxInformation[ReLU]={"ArgumentsPattern"->{}};
ForwardPropogateLayer[inputs_,ReLU]:=UnitStep[inputs-0]*inputs;
BackPropogateLayer[ReLU,inputs_,postLayerDeltaA_]:=
   postLayerDeltaA*UnitStep[inputs-0];
GradLayer[ReLU,layerInputs_,layerOutputDelta_]:={};
WeightDec[ReLU,grad_]:=ReLU;

(*Logistic*)
SyntaxInformation[Logistic]={"ArgumentsPattern"->{}};
ForwardPropogateLayer[inputs_,Logistic]:=1./(1.+Exp[-inputs]);
BackPropogateLayer[Logistic,inputs_,postLayerDeltaA_]:=
   postLayerDeltaA*Exp[inputs]*(1+Exp[inputs])^-2;
GradLayer[Logistic,layerInputs_,layerOutputDelta_]:={};
WeightDec[Logistic,grad_]:=Logistic;

(*PadFilterBank*)
SyntaxInformation[PadFilterBank]={"ArgumentsPattern"->{_}};
ForwardPropogateLayer[inputs_,PadFilterBank[padding_]]:=Map[ArrayPad[#,padding,.0]&,inputs,{2}]
BackPropogateLayer[PadFilterBank[padding_],postLayerDeltaA_]:=
   postLayerDeltaA[[All,All,padding+1;;-padding-1,padding+1;;-padding-1]];
GradLayer[PadFilterBank,layerInputs_,layerOutputDelta_]:={};
WeightDec[PadFilterBank[padding_],grad_]:=PadFilterBank[padding];

(* Ref: https://code.google.com/p/cuda-convnet/wiki/LayerParams# Local_response _normalization _layer _(same_map) *)
SyntaxInformation[RNorm]={"ArgumentsPattern"->{}};
RNorm\[Beta]=.00005;RNorm\[Alpha]=.75;
ForwardPropogateLayer[inputs_,RNorm]:=
   Map[(#*((1+(RNorm\[Alpha]/ListConvolve[ConstantArray[1.,{5,5}],ConstantArray[1.,#//Dimensions],{3,3},.0])*
      ListConvolve[ConstantArray[1.,{5,5}],#^2,{3,3},0.])^RNorm\[Beta])^-1)&,inputs,{2}];
BackPropogateLayer[RNorm,inputs_,postLayerDeltaA_]:=AbortAssert[0==1,"RNorm Backprop unimplemented."];

(*SubsampleFilterBankToFilterBank*)
SyntaxInformation[SubsampleFilterBankToFilterBank]={"ArgumentsPattern"->{}};
ForwardPropogateLayer[inputs_,SubsampleFilterBankToFilterBank]:=Map[#[[1;;-1;;2,1;;-1;;2]]&,inputs,{2}];
UpSample1[x_]:=Riffle[temp=Riffle[x,.0*x]//Transpose;temp,temp*.0]//Transpose;
BackPropogateLayer[SubsampleFilterBankToFilterBank,postLayerDeltaA_]:=
   Map[UpSample1,postLayerDeltaA,{2}];
GradLayer[SubsampleFilterBankToFilterBank,layerInputs_,layerOutputDelta_]:={};
WeightDec[SubsampleFilterBankToFilterBank,grad_]:=SubsampleFilterBankToFilterBank;

(*PadFilter*)
SyntaxInformation[PadFilter]={"ArgumentsPattern"->{_}};
ForwardPropogateLayer[inputs_,PadFilter[padding_]]:=Map[ArrayPad[#,padding,.0]&,inputs];
BackPropogateLayer[PadFilter[padding_],postLayerDeltaA_]:=
   postLayerDeltaA[[All,padding+1;;-padding-1,padding+1;;-padding-1]];
GradLayer[PadFilter,layerInputs_,layerOutputDelta_]:={};
WeightDec[PadFilter[padding_],grad_]:=PadFilter[padding];

(*MaxConvolveFilterBankToFilterBank*)
SyntaxInformation[MaxConvolveFilterBankToFilterBank]={"ArgumentsPattern"->{}};
ForwardPropogateLayer[inputs_,MaxConvolveFilterBankToFilterBank]:=Map[Max[Flatten[#]]&,
   Map[Partition[#,{3,3},{1,1},{-2,+2},-2.0]&,inputs,{2}],{4}];
BackPropogateLayer[MaxConvolveFilterBankToFilterBank,inputs_,outputs_,postLayerDeltaA_]:=(
   AbortAssert[Max[inputs]<1.4,"BackPropogateLayer::MaxConvolveFilterBankToFilterBank algo not designed for inputs > 1.4"];
(*   u1=Map[Partition[#,{3,3},{1,1},{-2,+2},-2.0]&,inputs,{2}];
   u2=Map[Max[Flatten[#]]&,u1,{4}];*)
   u3=Map[Partition[#,{3,3},{1,1},{-2,+2},1.5]&,outputs,{2}];
   u4=UnitStep[inputs-u3];
   u5=Map[Partition[#,{3,3},{1,1},{-2,+2},-2.0]&,postLayerDeltaA,{2}];
   u6=u4*u5;
   u7=Map[Total[Flatten[#]]&,u6,{4}])
GradLayer[MaxConvolveFilterBankToFilterBank,layerInputs_,layerOutputDelta_]:={};
WeightDec[MaxConvolveFilterBankToFilterBank,grad_]:=MaxConvolveFilterBankToFilterBank;


(* Some Test Helping Code *)
CheckGrad[lossF_,weight_,inputs_,targets_]:=
   (lossF[WeightDec[wl,-ReplacePart[gw*0.,weight->10^-8]],inputs,targets]-lossF[wl,inputs,targets])/10^-8

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


Size[net_,input_]:=(
   Print["# of Parameters ",Level[net,{-1}]//Length]; (*Slightly approx due to symbol vs function ie overcount Tanh etc *)
   Print["# of Neurons ",ForwardPropogateLayers[{input},net]//Flatten//Length];)

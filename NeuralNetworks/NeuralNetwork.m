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
ForwardPropogateLayers[inputs_,network_]:=
(* We don't include the inputs *)
   Rest[FoldList[LayerForwardPropogation,inputs,network]]

ForwardPropogate[inputs_,network_]:=
   ForwardPropogateLayers[inputs,network][[-1]]

BackPropogation[currentParameters_,inputs_,targets_,lossF_]:=(

   L = ForwardPropogateLayers[inputs, currentParameters];
   networkLayers=Length[currentParameters];

   AbortAssert[Dimensions[L[[networkLayers]]]==Dimensions[targets],"BackPropogation::Dimensions of outputs and targets should match"];

   DeltaL[networkLayers]=DeltaLoss[lossF,L[[networkLayers]],targets];

   For[layerIndex=networkLayers,layerIndex>1,layerIndex--,
(* layerIndex refers to layer being back propogated across
   ie computing delta's for layerIndex-1 given layerIndex *)

      Which[
         MatchQ[currentParameters[[layerIndex]],MaxPoolingFilterBankToFilterBank],
            DeltaL[layerIndex-1]=Backprop[currentParameters[[layerIndex]],L[[layerIndex-1]],L[[layerIndex]],DeltaL[layerIndex]];,
         MatchQ[currentParameters[[layerIndex]],Softmax],
            DeltaL[layerIndex-1]=Backprop[currentParameters[[layerIndex]],L[[layerIndex]],DeltaL[layerIndex]];,
         MatchQ[currentParameters[[layerIndex]],Tanh],
            DeltaL[layerIndex-1]=Backprop[currentParameters[[layerIndex]],L[[layerIndex-1]],DeltaL[layerIndex]];,
         MatchQ[currentParameters[[layerIndex]],Logistic],
            DeltaL[layerIndex-1]=Backprop[currentParameters[[layerIndex]],L[[layerIndex-1]],DeltaL[layerIndex]];,
         MatchQ[currentParameters[[layerIndex]],ReLU],
            DeltaL[layerIndex-1]=Backprop[currentParameters[[layerIndex]],L[[layerIndex-1]],DeltaL[layerIndex]];,
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

   Prepend[
      Table[
         LayerGrad[currentParameters[[layerIndex]],L[[layerIndex-1]],DeltaL[layerIndex]]
         ,{layerIndex,2,Length[currentParameters]}],
      LayerGrad[currentParameters[[1]],inputs,DeltaL[1]]
   ]
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
RegressionLoss1D[parameters_,inputs_,targets_]:=(outputs=ForwardPropogate[inputs,parameters];AbortAssert[Dimensions[outputs]==Dimensions[targets],"Loss1D::Mismatched Targets and Outputs"];Total[(outputs-targets)^2,2]/Length[inputs]);
RegressionLoss2D[parameters_,inputs_,targets_]:=Total[(ForwardPropogate[inputs,parameters]-targets)^2,3]/Length[inputs];
RegressionLoss3D[parameters_,inputs_,targets_]:=Total[(ForwardPropogate[inputs,parameters]-targets)^2,4]/Length[inputs];
ClassificationLoss[parameters_,inputs_,targets_]:=-Total[Log[Extract[ForwardPropogate[inputs,parameters],Position[targets,1]]]]/Length[inputs];

WeightDec[networkLayers_List,grad_List]:=MapThread[WeightDec,{networkLayers,grad}]

LineSearch[{\[Lambda]_,v_,current_},objectiveF_]:=
(* This is implicitly a lowest line search *)(
   t\[Lambda]=\[Lambda]*1.1; (*Has an optimism bias*)
   While[(loss=objectiveF[t\[Lambda]*v])>current,t\[Lambda]=t\[Lambda]*.5;AbortAssert[t\[Lambda]>10^-30]];
  {t\[Lambda],loss}
);

GenericGradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,algoF_,options_:{}]:=(
   trainingLoss=\[Infinity];
   validationLoss=\[Infinity];
   {validationInputs,validationTargets,maxEpoch,updateF,\[Lambda]} = {ValidationInputs,ValidationTargets,MaxEpoch,UpdateFunction,InitialLearningRate} /.
      options /. {ValidationInputs->{},ValidationTargets->{},MaxEpoch->20000,UpdateFunction->Identity,InitialLearningRate->.001};
   Print["Epoch: ",Dynamic[epoch]," Training Loss ",Dynamic[trainingLoss], " \[Lambda]=",Dynamic[\[Lambda]]];
   If[validationInputs!={},Print[" Validation Loss ",Dynamic[validationLoss]]];
   Print[Dynamic[grOutput]];
   For[wl=initialParameters;epoch=1,epoch<=maxEpoch,epoch++,
      algoF[];
      If[validationInputs!={},
        (*Note the validation can be done here as it is assumed it is small enough to compute in memory*)
         validationLoss=lossF[wl,validationInputs,validationTargets];
         AppendTo[ValidationHistory,validationLoss];,
         0];
      AppendTo[TrainingHistory,trainingLoss];
      updateF[];
   ]);

GradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,options_:{}]:=
   GenericGradientDescent[initialParameters,inputs,targets,gradientF,lossF,
      (  gw=gradientF[wl,inputs,targets,lossF];
         wl=WeightDec[wl,\[Lambda]*gw];
         trainingLoss=lossF[wl,inputs,targets];)&,
      options];

AdaptiveGradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,options_:{}]:=
   GenericGradientDescent[initialParameters,inputs,targets,gradientF,lossF,
      (  gw=gradientF[wl,inputs,targets,lossF];
         {\[Lambda],trainingLoss}=LineSearch[{\[Lambda],gw,trainingLoss},lossF[WeightDec[wl,#],inputs,targets]&];
         wl=WeightDec[wl,\[Lambda]*gw];
         trainingLoss=lossF[wl,inputs,targets];)&,
      options];

(*
AdaptiveGradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,options_:{}]:=(
   GenericGradientDescent[initialParameters,inputs,targets,gradientF,lossF,
      (gw=gradientF[wl,inputs,targets,lossF];
      {\[Lambda],trainingLoss}=LineSearch[{\[Lambda],gw,trainingLoss},Function[g,lossF[WeightDec[wl,g],inputs,targets]]];
      wl=WeightDec[wl,\[Lambda]*gw];)&,options];
   trainingLoss=lossF[wl,inputs,targets];)*)


MiniBatchGradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,options_:{}]:=(
   Print["Batch #:", Dynamic[batch], " Partial: ",Dynamic[partialTrainingLoss[[-1]]]];
   GenericGradientDescent[initialParameters,inputs,targets,gradientF,lossF,
      (  partialTrainingLoss={};batch=0;
         MapThread[
            (
            batch++;
            gw=gradientF[wl,#1,#2,lossF];
            wl=WeightDec[wl,\[Lambda]*gw];
            AppendTo[partialTrainingLoss,lossF[wl,#1,#2]];)&,
         {Partition[inputs,100],Partition[targets,100]}];
         trainingLoss = Mean[partialTrainingLoss])&,
      options];)


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
      If[ValidationHistory=={},TrainingHistory,{TrainingHistory,ValidationHistory}],
      PlotRange->All,PlotStyle->{Blue,Green}]);

WebMonitor[resourceName_]:=Function[{},
   Export[StringJoin[NNBaseDir,resourceName,".jpg"],
      Rasterize[{Text[trainingLoss],Text[validationLoss],ScreenMonitor[]},ImageSize->800,RasterSize->1000]];];

NNCheckpoint[resourceName_]:=Function[{},(WebMonitor[resourceName];Persist[resourceName])];


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
(* For below justification, see http://arxiv.org/pdf/1206.5533v2.pdf page 15 *)
FullyConnected1DTo1DInit[noFromNeurons_,noToNeurones_]:=
   FullyConnected1DTo1D[ConstantArray[0.,noToNeurones],Table[Random[]-.5,{noToNeurones},{noFromNeurons}]/Sqrt[noFromNeurons]]

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
ConvolveFilterBankToFilterBankInit[noOldFilterBank_,noNewFilterBank_,filterSize_]:=
   ConvolveFilterBankToFilterBank[
      Table[ConvolveFilterBankTo2D[0.,
            Table[Random[]-.5,{noOldFilterBank},{filterSize},{filterSize}]/Sqrt[noOldFilterBank*filterSize*filterSize]],
         {noNewFilterBank}]]

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

(*PadFilterBank*)
SyntaxInformation[PadFilterBank]={"ArgumentsPattern"->{_}};
LayerForwardPropogation[inputs_,PadFilterBank[padding_]]:=Map[ArrayPad[#,padding,.0]&,inputs,{2}]
Backprop[PadFilterBank[padding_],postLayerDeltaA_]:=
   postLayerDeltaA[[All,All,padding+1;;-padding-1,padding+1;;-padding-1]];
LayerGrad[PadFilterBank,layerInputs_,layerOutputDelta_]:={};
WeightDec[PadFilterBank[padding_],grad_]:=PadFilterBank[padding];


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

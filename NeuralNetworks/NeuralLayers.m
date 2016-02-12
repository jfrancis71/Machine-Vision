(* ::Package:: *)

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
Dropout - Implements dropout
*)


Needs["Developer`"];


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
BackPropogateLayer[FullyConnected1DTo1D[biases_,weights_],postLayerDeltaA_,_,_]:=postLayerDeltaA.weights
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
BackPropogateLayer[Convolve2D[biases_,weights_],postLayerDeltaA_,_,_]:=Table[ListConvolve[weights,postLayerDeltaA[[t]],{+1,-1},0],{t,1,Length[postLayerDeltaA]}]
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
BackPropogateLayer[Convolve2DToFilterBank[filters_],postLayerDeltaA_,inputs_,outputs_]:=Sum[BackPropogateLayer[filters[[f]],postLayerDeltaA[[All,f]],inputs,outputs],{f,1,Length[filters]}]
GradLayer[Convolve2DToFilterBank[filters_],layerInputs_,layerOutputDelta_]:=
   Table[{
         Total[layerOutputDelta[[All,filterIndex]],3],
         Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta[[All,filterIndex]],layerInputs}]]},
      {filterIndex,1,Length[filters]}];
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
BackPropogateLayer[FilterBankTo2D[bias_,weights_],postLayerDeltaA_,_,_]:=Transpose[Map[#*postLayerDeltaA&,weights],{2,1,3,4}]
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
BackPropogateLayer[FilterBankToFilterBank[biases_,weights_],postLayerDeltaA_,_,_]:=
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
BackPropogateLayer[Adaptor2DTo1D[width_],postLayerDeltaA_,_,_]:=
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
BackPropogateLayer[ConvolveFilterBankTo2D[bias_,kernels_],postLayerDeltaA_,_,_]:=(
   Transpose[Table[ListConvolve[{kernels[[w]]},postLayerDeltaA,{+1,-1},0],{w,1,Length[kernels]}],{2,1,3,4}]);
GradLayer[ConvolveFilterBankTo2D[bias_,kernels_],layerInputs_,layerOutputDelta_]:=(
   (*{Total[layerOutputDelta,3],Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta,layerInputs}]]}*)
   {Total[layerOutputDelta,3],Table[Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta,layerInputs[[All,w]]}]],{w,1,Length[kernels]}]})
WeightDec[networkLayer_ConvolveFilterBankTo2D,grad_]:=ConvolveFilterBankTo2D[networkLayer[[1]]-grad[[1]],networkLayer[[2]]-grad[[2]]];

(*ConvolveFilterBankToFilterBank*)
SyntaxInformation[ConvolveFilterBankToFilterBank]={"ArgumentsPattern"->{_}};
(* Ref: http://www.jimmyren.com/papers/aaai_vcnn.pdf
   On Vectorization of deep convolutional neural networks for vision tasks
   Jimmy Ren, Li Xu, 2015
*)
ForwardPropogateLayer[inputs_,ConvolveFilterBankToFilterBank[filters_]]:=(
   i2=If[installed,
         NNOverlapPartition[inputs,5],
         Map[Partition[#,{5,5},{1,1}]&,inputs,{2}]];
   i3=(Map[Flatten,
         Transpose[i2,{1,4,2,3,5,6}],{3}].
      Transpose[Map[Flatten,filters[[All,2]]]]);

   i4=Transpose[i3,{1,3,4,2}];
   Do[i4[[All,t]]=i4[[All,t]]+filters[[All,1]][[t]],{t,1,Length[i4[[1]]]}];i4
)
BackPropogateLayer[ConvolveFilterBankToFilterBank[filters_],postLayerDeltaA_,inputs_,outputs_]:=
   Sum[BackPropogateLayer[filters[[f]],postLayerDeltaA[[All,f]],inputs,outputs],{f,1,Length[filters]}];
GradLayer[ConvolveFilterBankToFilterBank[filters_],layerInputs_,layerOutputDelta_]:=
   Table[{
      Total[layerOutputDelta[[All,filterOutputIndex]],3],
      ListCorrelate[Transpose[{layerOutputDelta[[All,filterOutputIndex]]},{2,1,3,4}],layerInputs][[1]]},
      {filterOutputIndex,1,Length[filters]}]
WeightDec[ConvolveFilterBankToFilterBank[filters_],grad_]:=ConvolveFilterBankToFilterBank[WeightDec[filters,grad]];
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
BackPropogateLayer[MaxPoolingFilterBankToFilterBank,postLayerDeltaA_,layerInputs_,layerOutputs_]:=
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
BackPropogateLayer[Adaptor3DTo1D[features_,width_,height_],postLayerDeltaA_,_,_]:=
   unflatten[Flatten[postLayerDeltaA],{Length[postLayerDeltaA],features,width,height}];
GradLayer[Adaptor3DTo1D[features_,width_,height_],layerInputs_,layerOutputDelta_]:={};
WeightDec[networkLayer_Adaptor3DTo1D,grad_]:=Adaptor3DTo1D[networkLayer[[1]],networkLayer[[2]],networkLayer[[3]]];

(*Softmax*)
SyntaxInformation[Softmax]={"ArgumentsPattern"->{}};
ForwardPropogateLayer[inputs_,Softmax]:=Map[Exp[#]/Total[Exp[#]]&,inputs];
BackPropogateLayer[Softmax,postLayerDeltaA_,_,outputs_]:=
   Table[
      Sum[postLayerDeltaA[[n,i]]*outputs[[n,i]]*(KroneckerDelta[j,i]-outputs[[n,j]]),{i,1,Length[postLayerDeltaA[[1]]]}],
         {n,1,Length[postLayerDeltaA]},
      {j,1,Length[postLayerDeltaA[[1]]]}];
GradLayer[Softmax,layerInputs_,layerOutputDelta_]:={};
WeightDec[Softmax,grad_]:=Softmax;

(*Tanh*)
ForwardPropogateLayer[inputs_,Tanh]:=Tanh[inputs];
BackPropogateLayer[Tanh,postLayerDeltaA_,inputs_,_]:=
   postLayerDeltaA*Sech[inputs]^2;
GradLayer[Tanh,layerInputs_,layerOutputDelta_]:={};
WeightDec[Tanh,grad_]:=Tanh;

(*ReLU*)
SyntaxInformation[ReLU]={"ArgumentsPattern"->{}};
ForwardPropogateLayer[inputs_,ReLU]:=UnitStep[inputs-0]*inputs;
BackPropogateLayer[ReLU,postLayerDeltaA_,inputs_,_]:=
   postLayerDeltaA*UnitStep[inputs-0];
GradLayer[ReLU,layerInputs_,layerOutputDelta_]:={};
WeightDec[ReLU,grad_]:=ReLU;

(*Logistic*)
SyntaxInformation[Logistic]={"ArgumentsPattern"->{}};
ForwardPropogateLayer[inputs_,Logistic]:=1./(1.+Exp[-inputs]);
BackPropogateLayer[Logistic,postLayerDeltaA_,inputs_,_]:=
   postLayerDeltaA*Exp[inputs]*(1+Exp[inputs])^-2;
GradLayer[Logistic,layerInputs_,layerOutputDelta_]:={};
WeightDec[Logistic,grad_]:=Logistic;

(*PadFilterBank*)
SyntaxInformation[PadFilterBank]={"ArgumentsPattern"->{_}};
ForwardPropogateLayer[inputs_,PadFilterBank[padding_]]:=Map[ArrayPad[#,padding,.0]&,inputs,{2}]
BackPropogateLayer[PadFilterBank[padding_],postLayerDeltaA_,_,_]:=
   postLayerDeltaA[[All,All,padding+1;;-padding-1,padding+1;;-padding-1]];
GradLayer[PadFilterBank[padding_],layerInputs_,layerOutputDelta_]:={};
WeightDec[PadFilterBank[padding_],grad_]:=PadFilterBank[padding];

(* Ref: https://code.google.com/p/cuda-convnet/wiki/LayerParams# Local_response _normalization _layer _(same_map) *)
SyntaxInformation[RNorm]={"ArgumentsPattern"->{}};
RNorm\[Beta]=.00005;RNorm\[Alpha]=.75;
ForwardPropogateLayer[inputs_,RNorm]:=
   Map[(#*((1+(RNorm\[Alpha]/ListConvolve[ConstantArray[1.,{5,5}],ConstantArray[1.,#//Dimensions],{3,3},.0])*
      ListConvolve[ConstantArray[1.,{5,5}],#^2,{3,3},0.])^RNorm\[Beta])^-1)&,inputs,{2}];
BackPropogateLayer[RNorm,inputs_,postLayerDeltaA_,_,_]:=AbortAssert[0==1,"RNorm Backprop unimplemented."];

(*SubsampleFilterBankToFilterBank*)
SyntaxInformation[SubsampleFilterBankToFilterBank]={"ArgumentsPattern"->{}};
ForwardPropogateLayer[inputs_,SubsampleFilterBankToFilterBank]:=Map[#[[1;;-1;;2,1;;-1;;2]]&,inputs,{2}];
UpSample1[x_]:=Riffle[temp=Riffle[x,.0*x]//Transpose;temp,temp*.0]//Transpose;
BackPropogateLayer[SubsampleFilterBankToFilterBank,postLayerDeltaA_,_,_]:=
   Map[UpSample1,postLayerDeltaA,{2}];
GradLayer[SubsampleFilterBankToFilterBank,layerInputs_,layerOutputDelta_]:={};
WeightDec[SubsampleFilterBankToFilterBank,grad_]:=SubsampleFilterBankToFilterBank;

(*PadFilter*)
SyntaxInformation[PadFilter]={"ArgumentsPattern"->{_}};
ForwardPropogateLayer[inputs_,PadFilter[padding_]]:=Map[ArrayPad[#,padding,.0]&,inputs];
BackPropogateLayer[PadFilter[padding_],postLayerDeltaA_,_,_]:=
   postLayerDeltaA[[All,padding+1;;-padding-1,padding+1;;-padding-1]];
GradLayer[PadFilter[padding_],layerInputs_,layerOutputDelta_]:={};
WeightDec[PadFilter[padding_],grad_]:=PadFilter[padding];


(*MaxConvolveFilterBankToFilterBank*)
SyntaxInformation[MaxConvolveFilterBankToFilterBank]={"ArgumentsPattern"->{}};
installed=False;
ForwardPropogateLayer[inputs_,MaxConvolveFilterBankToFilterBank]:=
   If[installed,
      NNMaxListable[NNOverlapPartition[ArrayPad[inputs,{{0,0},{0,0},{1,1},{1,1}},-2.0],3]],
      Map[Max,Map[Partition[#,{3,3},{1,1},{-2,+2},-2.0]&,inputs,{2}],{4}]];

BackPropogateLayer[MaxConvolveFilterBankToFilterBank,postLayerDeltaA_,inputs_,outputs_]:=(
   AbortAssert[Max[inputs]<1.4,"BackPropogateLayer::MaxConvolveFilterBankToFilterBank algo not designed for inputs > 1.4"];
(*   u1=Map[Partition[#,{3,3},{1,1},{-2,+2},-2.0]&,inputs,{2}];
   u2=Map[Max[Flatten[#]]&,u1,{4}];*)
   Timer["MaxConvolveFilterBankToFilterBank::u3",u3=ToPackedArray[Map[Partition[#,{3,3},{1,1},{-2,+2},1.5]&,outputs,{2}]]];
   Timer["MaxConvolveFilterBankToFilterBank::u4",u4=UnitStep[inputs-u3]];
   Timer["MaxConvolveFilterBankToFilterBank::u5",u5=ToPackedArray[Map[Partition[#,{3,3},{1,1},{-2,+2},-2.0]&,postLayerDeltaA,{2}]]];
   Timer["MaxConvolveFilterBankToFilterBank::u6",u6=u4*u5];
   Timer["MaxConvolveFilterBankToFilterBank::u7",u7=Map[Total[Flatten[#]]&,u6,{4}]])
GradLayer[MaxConvolveFilterBankToFilterBank,layerInputs_,layerOutputDelta_]:={};
WeightDec[MaxConvolveFilterBankToFilterBank,grad_]:=MaxConvolveFilterBankToFilterBank;


(*
   Ref: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
*)
SyntaxInformation[DropoutLayer]={"ArgumentsPattern"->{_,_}};
SyntaxInformation[DropoutLayerMask]={"ArgumentsPattern"->{_}};
Dropout[layer_,inputs_]:=layer;
Dropout[net_List,inputs_]:=Map[Dropout[#,inputs]&,net];
Dropout[DropoutLayer[dims_,dropoutProb_],inputs_]:=
   DropoutLayerMask[Table[RandomInteger[],{Length[inputs]},dims]];
ForwardPropogateLayer[inputs_,DropoutLayer[_,_]]:=0.5*inputs;
ForwardPropogateLayer[inputs_,DropoutLayerMask[mask_]]:=inputs*mask;
BackPropogateLayer[DropoutLayerMask[mask_],postLayerDeltaA_,_,_]:=mask*postLayerDeltaA;
GradLayer[DropoutLayerMask[mask_],layerInputs_,layerOutputDelta_]:={};
GradLayer[DropoutLayer[_,_],layerInputs_,layerOutputDelta_]:={};
WeightDec[networkLayer_DropoutLayer,grad_]:=DropoutLayer[networkLayer[[1]],networkLayer[[2]]];


(*Entropy*)
(*
Original Example
samples=Table[mydata=RandomVariate[ParetoDistribution[1,3],100];1/100 Sum[Log[Abs[Nearest[mydata,mydata[[i]],20][[20]]-mydata[[i]]]],{i,1,100}] - PolyGamma[20] + PolyGamma[100] + Log[2],{1000}];
Ref: http://www.cs.tut.fi/~timhome/tim/tim/core/differential_entropy_kl_details.htm
Above link currently timing out
*)
NNEntropyList[samples_/;Length[samples]>=100]:=
   1/Length[samples] Sum[Log[Abs[Nearest[samples,samples[[i]],20][[20]]-samples[[i]]]],{i,1,Length[samples]}] - PolyGamma[20] + PolyGamma[100] + Log[2]

SyntaxInformation[NNEntropy]={"ArgumentsPattern"->{_}};
ForwardPropogateLayer[inputs_,NNEntropy[_]]:=(
   NNENT=Table[
      NNEntropyList[inputs[[All,f]]],
      {f,1,Length[inputs[[1]]]}];
   inputs)
BackPropogateLayer[NNEntropy[\[Lambda]_],postLayerDeltaA_,inputs_,_]:=
   postLayerDeltaA + Table[
         nr=Nearest[inputs[[All,f]],inputs[[ex,f]],20][[20]];
         diff=nr-inputs[[ex,f]];
         If[diff!=0,\[Lambda]*If[diff>0,-1,+1]*(1/Abs[diff]),0.],
      {ex,1,Length[inputs]},
      {f,1,Length[inputs[[1]]]}]
GradLayer[NNEntropy,layerInputs_,layerOutputDelta_]:={};
WeightDec[NNEntropy[\[Lambda]_],grad_]:=NNEntropy[\[Lambda]];

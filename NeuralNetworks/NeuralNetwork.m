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
   Z[0] = inputs;

   Module[{layerIndex=1},
   For[layerIndex=1,layerIndex<=Length[network],layerIndex++,
      A[layerIndex]=LayerForwardPropogation[Z[layerIndex-1],network[[layerIndex]]];

      Which[
         MatchQ[network[[layerIndex]],MaxPoolingFilterBankToFilterBank],
            Z[layerIndex]=A[layerIndex];,
         MatchQ[network[[layerIndex]],Softmax],
            Z[layerIndex]=A[layerIndex];,
         True,
            Z[layerIndex]=A[layerIndex];
      ];

   ];

   Z[layerIndex-1]]
)

BackPropogation[currentParameters_,inputs_,targets_,lossF_]:=(

   ForwardPropogation[inputs, currentParameters];
   networkLayers=Length[currentParameters];

   AbortAssert[Dimensions[Z[networkLayers]]==Dimensions[targets],"BackPropogation::Dimensions of outputs and targets should match"];

   DeltaZ[networkLayers]=DeltaLoss[lossF,Z[networkLayers],targets];
   DeltaA[networkLayers]=DeltaZ[networkLayers]*Sech[A[networkLayers]]^2;

   For[layerIndex=networkLayers,layerIndex>1,layerIndex--,
(* layerIndex refers to layer being back propogated across
   ie computing delta's for layerIndex-1 given layerIndex *)
      Which[
         MatchQ[currentParameters[[layerIndex]],MaxPoolingFilterBankToFilterBank],
            DeltaA[layerIndex-1]=Backprop[currentParameters[[layerIndex]],Z[layerIndex-1],A[layerIndex],DeltaA[layerIndex]];,
         MatchQ[currentParameters[[layerIndex]],Softmax],
            DeltaA[layerIndex-1]=Backprop[currentParameters[[layerIndex]],Z[layerIndex-1],DeltaZ[layerIndex]];,
         True,
            (DeltaZ[layerIndex-1]=Backprop[currentParameters[[layerIndex]],DeltaA[layerIndex]];
             DeltaA[layerIndex-1]=DeltaZ[layerIndex-1]*Sech[A[layerIndex-1]]^2)
      ];
   ];
)

(*
   The linear activation layer has shape T*U 
   DeltaXX refers to the partial derivative of the loss function wrt that neurone activation
      so it has shape T*U
   targets has shape T*O where O is the number of output units
*)
Grad[currentParameters_,inputs_,targets_,lossF_]:=(

   AbortAssert[Length[inputs]==Length[targets],"Grad::# of Training Labels should equal # of Training Inputs"];

   BackPropogation[currentParameters,inputs,targets,lossF];

   Table[
      LayerGrad[currentParameters[[layerIndex]],Z[layerIndex-1],DeltaA[layerIndex]]
      ,{layerIndex,1,Length[currentParameters]}]
)

DeltaLoss[RegressionLoss1D,outputs_,targets_]:=2.0*(outputs-targets);
DeltaLoss[RegressionLoss2D,outputs_,targets_]:=2.0*(outputs-targets);
DeltaLoss[RegressionLoss3D,outputs_,targets_]:=2.0*(outputs-targets);
DeltaLoss[ClassificationLoss,outputs_,targets_]:=-1/Extract[outputs,Position[targets,1]];
DeltaLoss[ClassificationLoss,outputs_,targets_]:=-targets*(1.0/outputs);

(*This is implicitly a regression loss function*)
RegressionLoss1D[parameters_,inputs_,targets_]:=(outputs=ForwardPropogation[inputs,parameters];AbortAssert[Dimensions[outputs]==Dimensions[targets],"Loss1D::Mismatched Targets and Outputs"];Total[(outputs-targets)^2,2]/Length[inputs]);
RegressionLoss2D[parameters_,inputs_,targets_]:=Total[(ForwardPropogation[inputs,parameters]-targets)^2,3]/Length[inputs];
RegressionLoss3D[parameters_,inputs_,targets_]:=Total[(ForwardPropogation[inputs,parameters]-targets)^2,4]/Length[inputs];
ClassificationLoss[parameters_,inputs_,targets_]:=-Total[Log[Extract[ForwardPropogation[inputs,parameters],Position[targets,1]]]]/Length[inputs];

WeightDec[networkLayers_List,grad_List]:=MapThread[WeightDec,{networkLayers,grad}]

GradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,\[Lambda]_,maxLoop_:2000]:=(
   Print["Iter: ",Dynamic[loop],"Current Loss", Dynamic[loss]];
   For[wl=initialParameters;loop=1,loop<=maxLoop,loop++,PreemptProtect[loss=lossF[wl,inputs,targets];wl=WeightDec[wl,(gw=\[Lambda]*gradientF[wl,inputs,targets])]]];
   wl )

AdaptiveGradientDescent[initialParameters_,inputs_,targets_,gradientF_,lossF_,options_:{}]:=(
   \[Lambda]=.000001;
   trainingLoss=-\[Infinity];
   {validationInputs,validationTargets,maxLoop} = {ValidationInputs,ValidationTargets,MaxLoop} /.
      options /. {ValidationInputs->{},ValidationTargets->{},MaxLoop->20000};
   Print["Iter: ",Dynamic[loop]," Training Loss ",Dynamic[trainingLoss], " \[Lambda]=",Dynamic[\[Lambda]]];
   If[validationInputs!={},Print[" Validation Loss ",Dynamic[validationLoss]]];
   For[wl=initialParameters;loop=1,loop<=maxLoop,loop++,
      trainingLoss=lossF[wl,inputs,targets];
      If[validationInputs!={},validationLoss=lossF[wl,validationInputs,validationTargets],validationLoss=0.0];
      twl=WeightDec[wl,gw=\[Lambda] gradientF[wl,inputs,targets,lossF]];
      If[lossF[twl,inputs,targets]<lossF[wl,inputs,targets],(wl=twl;\[Lambda]=\[Lambda]*2),(\[Lambda]=\[Lambda]*0.5)];
])

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

   Map[ListCorrelate[layerKernel,#]&,inputs]+layerBias
)
Backprop[Convolve2D[biases_,weights_],postLayerDeltaA_]:=Table[ListConvolve[weights,postLayerDeltaA[[t]],{+1,-1}],{t,1,Length[postLayerDeltaA]}]
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
   bias+Table[Sum[ListCorrelate[kernels[[kernel]],input[[kernel]]],{kernel,1,Length[kernels]}],{input,inputs}]);
Backprop[ConvolveFilterBankTo2D[bias_,kernels_],postLayerDeltaA_]:=(
   (*Table[Print[postLayerDeltaA[[input]]//Dimensions].Print[kernel//Dimensions],{input,1,Length[postLayerDeltaA]},{kernel,kernels}]);*)
   Table[ListConvolve[kernels[[w]],postLayerDeltaA[[t]],{+1,-1}],{t,1,Length[postLayerDeltaA]},{w,1,Length[kernels]}])
LayerGrad[ConvolveFilterBankTo2D[bias_,kernels_],layerInputs_,layerOutputDelta_]:=(
   (*{Total[layerOutputDelta,3],Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta,layerInputs}]]}*)
   {Total[layerOutputDelta,3],Table[Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta,layerInputs[[All,w]]}]],{w,1,Length[kernels]}]})
WeightDec[networkLayer_ConvolveFilterBankTo2D,grad_]:=ConvolveFilterBankTo2D[networkLayer[[1]]-grad[[1]],networkLayer[[2]]-grad[[2]]];

(*ConvolveFilterBankToFilterBank*)
SyntaxInformation[ConvolveFilterBankToFilterBank]={"ArgumentsPattern"->{_}};
LayerForwardPropogation[inputs_,ConvolveFilterBankToFilterBank[filters_]]:=(
   Transpose[Map[LayerForwardPropogation[inputs,#]&,filters],{2,1,3,4}]
);
Backprop[ConvolveFilterBankToFilterBank[filters_],postLayerDeltaA_]:=Sum[Backprop[filters[[f]],postLayerDeltaA[[All,f]]],{f,1,Length[filters]}];
LayerGrad[ConvolveFilterBankToFilterBank[filters_],layerInputs_,layerOutputDelta_]:=
   Table[{Total[layerOutputDelta[[All,filterOutputIndex]],3],Table[Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta[[All,filterOutputIndex]],layerInputs[[All,l]]}]],{l,1,Length[layerInputs[[1]]]}]},{filterOutputIndex,1,Length[filters]}]
WeightDec[networkLayer_ConvolveFilterBankToFilterBank,grad_]:=ConvolveFilterBankToFilterBank[WeightDec[networkLayer[[1]],grad]];

(*MaxPoolingFilterBankToFilterBank*)
SyntaxInformation[MaxPoolingFilterBankToFilterBank]={"ArgumentsPattern"->{}};
LayerForwardPropogation[inputs_,MaxPoolingFilterBankToFilterBank]:=
   Map[Function[image,Map[Max,Partition[image,{2,2}],{2}]],inputs,{2}];
backRouting[previousZ_,nextA_]:=UnitStep[Map[Partition[#,{2,2}]&,previousZ,{2}]-
   Map[Function[image,Map[{{#,#},{#,#}}&,image,{2}]],nextA,{2}]];
Backprop[MaxPoolingFilterBankToFilterBank,layerInputs_,layerOutputs_,postLayerDeltaA_]:=
   Map[ArrayFlatten,backRouting[layerInputs,layerOutputs]*Map[{{#,#},{#,#}}&,postLayerDeltaA,{4}],4];
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
Backprop[Softmax,inputs_,postLayerDeltaA_]:=
   Table[
      Sum[postLayerDeltaA[[n,i]]*((If[i==j,Exp[inputs[[n,i]]],0] Total[Exp[inputs[[n]]]]) - (Exp[inputs[[n,j]]]*Exp[inputs[[n,i]]]))/(Total[Exp[inputs[[n]]]])^2,{i,1,Length[postLayerDeltaA[[1]]]}],
   {n,1,Length[postLayerDeltaA]},{j,1,Length[postLayerDeltaA[[1]]]}]
LayerGrad[Softmax,layerInputs_,layerOutputDelta_]:={};
WeightDec[Softmax,grad_]:=Softmax;


(* Examples *)
sqNetwork={
   FullyConnected1DTo1D[{.2,.3},{{2},{3}}],
   FullyConnected1DTo1D[{.6},{{1,7}}]
};
sqInputs=Transpose[{Table[x,{x,-1,1,0.1}]}];sqInputs//MatrixForm;
sqOutputs=sqInputs^2;sqOutputs//MatrixForm;
sqTrain:=GradientDescent[sqNetwork,sqInputs,sqOutputs,Grad,RegressionLoss1D,.0001,500000];


XORNetwork={
   FullyConnected1DTo1D[{.2,.3},{{2,.3},{1,Random[]-.5}}],
   FullyConnected1DTo1D[{.6},{{1,Random[]-.5}}]
};
XORInputs={{0,0},{0,1},{1,0},{1,1}};XORInputs//MatrixForm;
XOROutputs=Transpose[{{0,1,1,0}}];XOROutputs//MatrixForm;
XORTrain:=GradientDescent[XORNetwork,XORInputs,XOROutputs,Grad,RegressionLoss1D,.0001,500000];


MultInputs=Flatten[Table[{a,b},{a,0,1,.1},{b,0,1,.1}],1];MultInputs//MatrixForm;
MultOutputs=Map[{#[[1]]*#[[2]]}&,MultInputs];MultOutputs//MatrixForm;
MultTrain:=GradientDescent[XORNetwork,MultInputs,MultOutputs,Grad,RegressionLoss1D,.0001,5000000];


edgeNetwork={Convolve2D[0,Table[Random[],{3},{3}]]};
edgeInputs={StandardiseImage["C:\\Users\\Julian\\Google Drive\\Personal\\Pictures\\Dating Photos\\me3.png"]};
edgeOutputs=ForwardPropogation[edgeInputs,{Convolve2D[0,sobelY]}];
edgeTrain:=GradientDescent[edgeNetwork,edgeInputs,edgeOutputs,Grad,RegressionLoss2D,.000001,500000]


edgeFilterBankNetwork={Convolve2DToFilterBank[{Convolve2D[0,Table[Random[],{3},{3}]],Convolve2D[0,Table[Random[],{3},{3}]]}]};
edgeFilterBankOutputs=ForwardPropogation[edgeInputs,{Convolve2DToFilterBank[{Convolve2D[0,sobelY],Convolve2D[0,sobelX]}]}];
edgeFilterBankTrain:=GradientDescent[edgeFilterBankNetwork,edgeInputs,edgeFilterBankOutputs,Grad,RegressionLoss3D,.000001,500000]


edgeFilterBankTo2DNetwork={FilterBankTo2D[.3,{.3,.5}]};
edgeFilterBankTo2DInputs=edgeFilterBankOutputs;
edgeFilterBankTo2DOutputs=edgeOutputs;
edgeFilterBankTo2DTrain:=GradientDescent[edgeFilterBankTo2DNetwork,edgeFilterBankTo2DInputs,edgeFilterBankTo2DOutputs,Grad,RegressionLoss2D,.000001,500000]


Deep1Network=Join[edgeFilterBankNetwork,edgeFilterBankTo2DNetwork];
Deep1Inputs=edgeInputs;
Deep1Outputs=edgeOutputs;
Deep1Train:=GradientDescent[Deep1Network,Deep1Inputs,Deep1Outputs,Grad,RegressionLoss2D,.000001,500000];
Deep1Monitor:=Dynamic[{wl[[2,2]],{Show[Deep1Network[[1,1,1,2]]//ColDispImage,ImageSize->35],Show[Deep1Network[[1,1,2,2]]//ColDispImage,ImageSize->35]},
{{-gw[[1,1,2]]//MatrixForm,-gw[[1,2,2]]//MatrixForm},-gw[[2,2]]}
}]


Deep2Network=Join[edgeFilterBankNetwork,edgeFilterBankTo2DNetwork];
Deep2Inputs=edgeInputs;
Deep2Outputs=(ForwardPropogation[edgeInputs,{Convolve2D[0,sobelX]}]+ForwardPropogation[edgeInputs,{Convolve2D[0,sobelY]}])/2;
Deep2Train:=AdaptiveGradientDescent[Deep2Network,Deep2Inputs,Deep2Outputs,Grad,RegressionLoss2D,{MaxLoss->500000}];
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
SemTrain:=GradientDescent[SemNetwork,SemInputs,SemOutputs,Grad,RegressionLoss1D,.0001,500000];


SeedRandom[1234];
r1=Partition[RandomReal[{0,1},9],3];r2=Partition[RandomReal[{0,1},9],3];
r3=RandomReal[{0,1},4];
r4=Partition[RandomReal[{0,1},8],2];
r5=Random[];r6=RandomReal[{0,1},4];
FTBNetwork={
   Convolve2DToFilterBank[{Convolve2D[0.,r1-.5],Convolve2D[0.,r2-.5]}],
   FilterBankToFilterBank[0.,r4-.5],
   FilterBankTo2D[0.,r6-.5]};
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
   FilterBankToFilterBank[{.4,.7,.1,.7},{{.3,.2},{.7,.3},{.15,.16},{.32,.31}}],
   FilterBankTo2D[.3,{.3,.5,.1,.2}]};
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
      Convolve2D[0,Partition[RandomReal[{0,1},9]-.5,3]]}],
   ConvolveFilterBankToFilterBank[{
     ConvolveFilterBankTo2D[0,{Partition[RandomReal[{0,1},9]-.5,3],Partition[RandomReal[{0,1},9]-.5,3]}],
     ConvolveFilterBankTo2D[0,{Partition[RandomReal[{0,1},9]-.5,3],Partition[RandomReal[{0,1},9]-.5,3]}],
     ConvolveFilterBankTo2D[0,{Partition[RandomReal[{0,1},9]-.5,3],Partition[RandomReal[{0,1},9]-.5,3]}]
}],
   FilterBankTo2D[0.,{.1,.6,.2}]   
};
TestConvolveInputs=edgeInputs/4;
TestConvolveOutputs=edgeInputs[[All,3;;-3,3;;-3]]/4;
TestConvolveTrain:=AdaptiveGradientDescent[TestConvolveNetwork,TestConvolveInputs,TestConvolveOutputs,Grad,RegressionLoss2D,{MaxLoop->500000}];


SeedRandom[1234];
TestMaxNetwork={
   Convolve2DToFilterBank[{
      Convolve2D[0,Partition[RandomReal[{0,1},9]-.5,3]],
      Convolve2D[0,Partition[RandomReal[{0,1},9]-.5,3]]}],
   MaxPoolingFilterBankToFilterBank,
   FilterBankTo2D[0,{.1,.5}]
};
TestMaxInputs=edgeInputs[[All,1;;-2,All]]/4;
TestMaxOutputs=Table[.25*Max[
   edgeInputs[[t,y*2,x*2-1]],
   edgeInputs[[t,y*2,x*2]],
   edgeInputs[[t,y*2-1,x*2-1]],
   edgeInputs[[t,y*2-1,x*2]]]
   ,{t,1,1},{y,1,70},{x,1,63}];
TestMaxTrain:=AdaptiveGradientDescent[TestMaxNetwork,TestMaxInputs,TestMaxOutputs,Grad,RegressionLoss2D,{MaxLoop->500000}];

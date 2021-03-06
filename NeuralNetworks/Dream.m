(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


DeltaLoss[DreamLoss,outputs_,targets_]:=targets*1.;


DreamLoss[parameters_,inputs_,targets_]:=Total[Extract[ForwardPropogate[inputs,parameters],Position[targets,1]]]/Length[inputs];


Dream::usage = "Dream[inputDims,neuron] returns input which maximises neuron response.\n
Note if neuron is saturated, then gradients may have difficulty propogating and optimisation will be exceptionally slow";


Options[Dream] = { MaxIterations -> 2000, InitialLearningRate->.01 };


Dream[inputDims_,neuron_,opts:OptionsPattern[]]:=(
   dream=Array[.5&,inputDims];
   neuronLayer=neuron[[1]];
   target=ReplacePart[ForwardPropogate[{dream},wl[[1;;neuronLayer]]][[1]]*.0,Rest[neuron]->1.0];
   dream=GradientDescent[dream,(
      L=ForwardPropogateLayers[{#},wl[[1;;neuronLayer]]];
      deltan=BackPropogateLayers[wl[[1;;neuronLayer]],L,DeltaLoss[DreamLoss,L[[-1]],{target}]];
      dw=BackPropogateLayer[wl[[1]],deltan[[1]],_,_];First[dw])&,(#1-#2)&,opts];
   dream
)


Salient[f_,image_]:=(
   L=ForwardPropogateLayers[{image},wl[[1;;-1]]];
   deltan=BackPropogateLayers[wl[[1;;-1]],L,DeltaLoss[DreamLoss,L[[-1]],{f}]];
   dw=BackPropogateLayer[wl[[1]],deltan[[1]],_,_][[1]];
   Abs[dw]/Max[Abs[dw]]
)

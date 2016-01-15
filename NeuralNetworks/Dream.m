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
      BackPropogation[wl[[1;;neuronLayer]],{#},{target},DreamLoss];
      dw=BackPropogateLayer[wl[[1]],DeltaL[1]];First[dw])&,(#1-#2)&,opts];
   dream
)


Salient[f_,image_]:=(
   BackPropogation[wl[[1;;-1]],{image},{f},DreamLoss];
   dw=BackPropogateLayer[wl[[1]],DeltaL[1]][[1]];
   dw/Max[dw]
)

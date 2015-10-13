(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"
<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/MNIST/MNISTData.m"


NNRead["MNIST\\LeNet1"];


DeltaLoss[DreamLoss,outputs_,targets_]:=targets*1.;


DreamLoss[parameters_,inputs_,targets_]:=Total[Extract[ForwardPropogate[inputs,parameters],Position[targets,1]]]/Length[inputs];


Dream::usage = "Dream[inputDims,neuron] returns input which maximises neuron response.\n
Note if neuron is saturated, then gradients may have difficulty propogating and optimisation will be exceptionally slow";


Options[Dream] = { MaxIterations -> 2000 };


Dream[inputDims_,neuron_,opts___]:=(
   dream=Array[.5&,inputDims];
   neuronLayer=neuron[[1]];
   target=ReplacePart[ForwardPropogate[{dream},wl[[1;;neuronLayer]]][[1]]*.0,Rest[neuron]->1.0];
   For[sl=0,sl<=(MaxIterations/.{opts}/.Options[Dream]),sl++,
      BackPropogation[wl[[1;;neuronLayer]],{dream},{target},DreamLoss];
      dw=BackPropogateLayer[wl[[1]],DeltaL[1]];
      fw=UnitStep[dream-1.];
      ef=UnitStep[-dream];
      (*dream += (.01*dw[[1]]/Max[dw[[1]]]-fw+ef)*1.0*10^-1;*)
      dream += .01*dw[[1]];
      (*dream = dream/Norm[dream];*)
      (*Clip[dream,{0.,1.}];*)
      error=DreamLoss[wl[[1;;-1]],{dream},{target}];
];
   dream
)


Salient[f_,image_]:=(
   BackPropogation[wl[[1;;-1]],{image},{f},DreamLoss];
   dw=BackPropogateLayer[wl[[1]],DeltaL[1]][[1]];
   dw/Max[dw]
)

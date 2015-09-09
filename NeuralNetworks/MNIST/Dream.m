(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"
<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/MNIST/MNISTData.m"


NNRead["MNIST\\LeNet1"];


DeltaLoss[DreamLoss,outputs_,targets_]:={target*1.};


DreamLoss[parameters_,inputs_,targets_]:=Total[Extract[ForwardPropogate[inputs,parameters],Position[targets,1]]]/Length[inputs];


Dream[f_]:=(
   target=ReplacePart[ConstantArray[0,10],f->1];
   dream=Table[Random[],{20},{20}];
   For[sl=0,sl<=2000,sl++,
      BackPropogation[wl[[1;;-1]],{dream},{target},DreamLoss];
      dw=Backprop[wl[[1]],DeltaL[1]];
      fw=UnitStep[dream-1.];
      ef=UnitStep[-dream];
      dream += (.01*dw[[1]]/Max[dw[[1]]]-fw+ef)*1.0*10^-1;
      (*Clip[dream,{0.,1.}];*)
      error=DreamLoss[wl[[1;;-1]],{dream},{target}];
];
   dream
)

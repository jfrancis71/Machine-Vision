(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


AddNoise[inputs_]:=(
   tmp1=UnitStep[RandomReal[{0,1},inputs//Dimensions]-.667];
   (inputs*(1-tmp1)+tmp1*RandomReal[{0,1},inputs//Dimensions])
)


NoisyTiedGrad[wl_,inputs_,targets_,lossF_]:=
   TiedGrad[wl,AddNoise[inputs],targets,lossF];

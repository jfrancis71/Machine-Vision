(* ::Package:: *)

(*Autoencoder
Network achieves good encoding
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


AutoencoderNetwork={
   FullyConnected1DTo1D[
      ConstantArray[0,32],Table[Random[],{32},{32}]-.5],
   FullyConnected1DTo1D[
      ConstantArray[0,5],Table[Random[],{5},{32}]-.5],
   FullyConnected1DTo1D[
      ConstantArray[0,32],Table[Random[],{32},{5}]-.5],
   FullyConnected1DTo1D[
      ConstantArray[0,32],Table[Random[],{32},{32}]-.5]
};


AutoencoderInputs=Table[ReplacePart[ConstantArray[0,32],{t->1,33-t->1}],{t,1,32}];


AutoencoderOutputs=AutoencoderInputs;


wl=AutoencoderNetwork;


AutoencoderTrained:=AdaptiveGradientDescent[wl,AutoencoderInputs,AutoencoderOutputs,Grad,Loss2D,{MaxLoop->500000}];

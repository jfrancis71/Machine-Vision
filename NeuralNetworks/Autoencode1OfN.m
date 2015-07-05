(* ::Package:: *)

(*Autoencoder
Network achieves good encoding
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


H=64;
B=1;
Autoencoder1OfNNetwork={
   FullyConnected1DTo1D[
      ConstantArray[0,H],Table[Random[],{H},{32}]-.5],
   FullyConnected1DTo1D[
      ConstantArray[0,B],Table[Random[],{B},{H}]-.5],
   FullyConnected1DTo1D[
      ConstantArray[0,H],Table[Random[],{H},{B}]-.5],
   FullyConnected1DTo1D[
      ConstantArray[0,32],Table[Random[],{32},{H}]-.5]
};


AutoencoderInputs=Table[ReplacePart[ConstantArray[0,32],t->1],{t,1,32}];


AutoencoderOutputs=AutoencoderInputs;


wl=Autoencoder1OfNNetwork;


Autoencode1OfNTrained:=AdaptiveGradientDescent[wl,AutoencoderInputs,AutoencoderOutputs,Grad,Loss2D,{MaxLoop->500000}];


(* Achieves .625 *)
H1=64;
H2=10;
B=1;
Autoencoder1OfNNetworkA2={
   FullyConnected1DTo1D[
      ConstantArray[0,H1],Table[Random[],{H1},{32}]-.5],
   FullyConnected1DTo1D[
      ConstantArray[0,H2],Table[Random[],{H2},{H1}]-.5],
   FullyConnected1DTo1D[
      ConstantArray[0,B],Table[Random[],{B},{H2}]-.5],
   FullyConnected1DTo1D[
      ConstantArray[0,H2],Table[Random[],{H2},{B}]-.5],
   FullyConnected1DTo1D[
      ConstantArray[0,H1],Table[Random[],{H1},{H2}]-.5],
   FullyConnected1DTo1D[
      ConstantArray[0,32],Table[Random[],{32},{H1}]-.5]
};


wl=Autoencoder1OfNNetworkA2;


Autoencode1OfNTrained:=AdaptiveGradientDescent[wl,AutoencoderInputs,AutoencoderOutputs,Grad,Loss2D,{MaxLoop->500000}];

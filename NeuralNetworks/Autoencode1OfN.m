(* ::Package:: *)

(*
   Autoencoder
   Encodes 1 Of N pattern
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


(* Achieves .625 *)
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


Autoencode1OfNTrain:=AdaptiveGradientDescent[wl,AutoencoderInputs,AutoencoderOutputs,Grad,Loss2D,{MaxLoop->500000}];


 (* Achieves  .531134 after 20 hours *)

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


(* Achieves good encoding after few minutes
*)
Autoencoder1OfNNetworkA3={
   FullyConnected1DTo1D[
      ConstantArray[0,32],Table[Random[],{32},{32}]-.5],
   FullyConnected1DTo1D[
      ConstantArray[0,5],Table[Random[],{5},{32}]-.5],
   FullyConnected1DTo1D[
      ConstantArray[0,32],Table[Random[],{32},{5}]-.5],
   FullyConnected1DTo1D[
      ConstantArray[0,32],Table[Random[],{32},{32}]-.5]
};


wl=Autoencoder1OfNNetworkA3;


(* Achieves good encoding in few minutes, about 50,000 iterations. Loss .0136
*)
H1=96;
B=3;
R1=(Table[Random[],{H1},{32}]-.5)/H1;
R2=Table[Random[],{B},{H1}]-.5;
Autoencoder1OfNNetworkA4={
   FullyConnected1DTo1D[
      ConstantArray[0,H1],R1],
   FullyConnected1DTo1D[
      ConstantArray[0,B],R2],
   FullyConnected1DTo1D[
      ConstantArray[0,H1],Transpose[R2]],
   FullyConnected1DTo1D[
      ConstantArray[0,32],Transpose[R1]]
};


wl=Autoencoder1OfNNetworkA4;


Autoencode1OfNTrain:=AdaptiveGradientDescent[wl,AutoencoderInputs,AutoencoderOutputs,Grad,Loss2D,{MaxLoop->500000}];


(*
Achieves good encoding using intermediate eventually
  takes a couple of days, parameters not currently in version control
  interpretation of decoding hidden layer not obvious
*)


AutoencoderIntermediate=Table[{s/16}-1,{s,1,32}]//N


H1=96;
B=1;
R1=(Table[Random[],{H1},{32}]-.5)/H1;
R2=Table[Random[],{B},{H1}]-.5;
Autoencoder1OfNNetworkA5P1={
   FullyConnected1DTo1D[
      ConstantArray[0,H1],R1],
   FullyConnected1DTo1D[
      ConstantArray[0,B],R2]
   };
H1=96;
B=1;
R1=(Table[Random[],{H1},{32}]-.5)/H1;
R2=Table[Random[],{B},{H1}]-.5;
Autoencoder1OfNNetworkA5P2={

      FullyConnected1DTo1D[
      ConstantArray[0,H1],Transpose[R2]],
   FullyConnected1DTo1D[
      ConstantArray[0,32],Transpose[R1]]
};


wlP1=Autoencoder1OfNNetworkA5P1;
wlP2=Autoencoder1OfNNetworkA5P2;


Autoencode1OfNTrainP1:=AdaptiveGradientDescent[wlP1,AutoencoderInputs,AutoencoderIntermediate,Grad,Loss2D,{MaxLoop->500000}];
Autoencode1OfNTrainP2:=AdaptiveGradientDescent[wlP2,AutoencoderIntermediate,AutoencoderOutputs,Grad,Loss2D,{MaxLoop->10^7}];

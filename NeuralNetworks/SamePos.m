(* ::Package:: *)

(*
Challenge Same Pos problem
Achieve loss 0.0003576527158468417
Takes 3 hours where not much happens then gets good representation
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


SamePos1Network={
   FullyConnected1DTo1D[
      ConstantArray[0,2],(Table[Random[],{2},{64}]-.5)/64.],
   FullyConnected1DTo1D[
      ConstantArray[0,2],Table[Random[],{2},{2}]-.5],
   FullyConnected1DTo1D[
      ConstantArray[0,1],Table[Random[],{1},{2}]-.5]
};


SamePos2Network={
   FullyConnected1DTo1D[
      ConstantArray[0,32],(Table[Random[],{32},{64}]-.5)/64.],
   FullyConnected1DTo1D[
      ConstantArray[0,1],Table[Random[],{1},{32}]-.5]
};


SamePosInputs=Flatten[Table[Join[ReplacePart[ConstantArray[0,32],i->1],ReplacePart[ConstantArray[0,32],j->1]],{i,1,32},{j,1,32}],1];


SamePosOutputs=Map[{Boole[(Partition[#,32][[1]]
==
Partition[#,32][[2]])]}&,SamePosInputs];


wl=SamePos1Network;


SamePos1Train:=AdaptiveGradientDescent[wl,SamePosInputs,SamePosOutputs,Grad,Loss2D,{MaxLoop->500000}];


wl=SamePos2Network;


SamePos2Train:=AdaptiveGradientDescent[wl,SamePosInputs,SamePosOutputs,Grad,Loss2D,{MaxLoop->500000}];

(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


(*
  First state entry is visible vector
 Second entry is hidden vector
Weight vector is H*V
*)
RBMIter[state_,{biases_,weights_}]:=( 
hidden=LogisticFn[Flatten[weights.Transpose[{(state[[1]])}]+biases[[2]]]];
hidden=Map[Boole[RandomReal[]<#]&,hidden];
vis1=LogisticFn[Flatten[Transpose[weights].Transpose[{(hidden)}]+biases[[1]]]];
vis=Map[Boole[RandomReal[]<#]&,vis1];
{vis,hidden})


init={{1,1},{1}};


biasInit={{-3,+3},{0}};


weightInit={{6,-6}};


RBM[state_,param_,iterations_]:=
NestList[RBMIter[#,param]&,state,iterations]


example=RBM[
init,
{biasInit,weightInit
},100][[All,1]]//DispImage

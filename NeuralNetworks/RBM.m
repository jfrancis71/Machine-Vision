(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


RBMSampleHidden[visible_,{biases_,weights_}]:=(
   hidden=LogisticFn[Flatten[weights.Transpose[{(visible)}]+biases[[2]]]];
   hidden=Map[Boole[RandomReal[]<#]&,hidden])



RBMSampleVisible[hidden_,{biases_,weights_}]:=(
   vis1=LogisticFn[Flatten[Transpose[weights].Transpose[{(hidden)}]+biases[[1]]]];
   vis=Map[Boole[RandomReal[]<#]&,vis1])


(*
  First state entry is visible vector
 Second entry is hidden vector
Weight vector is H*V
*)
RBMIter[state_,{biases_,weights_}]:=( 
   hidden = RBMSampleHidden[state[[1]],{biases,weights}];
   vis = RBMSampleVisible[hidden,{biases,weights}];
   {vis,hidden})


init={{1,1},{1}};


biasInit={{-3,+3},{0}};


weightInit={{6,-6}};


RBM[state_,param_,iterations_]:=
NestList[RBMIter[#,param]&,state,iterations]


mynet = {biasInit,weightInit};


example=RBM[init,mynet,100][[All,1]];example//DispImage;


(*
   Ref: https://www.youtube.com/watch?v=wMb7cads0go
   Hugo Larochelle Neural Networks 5.5 minutes 8:14
*)
RBMTrain[net_,examples_]:=For[wl=net;l=0,
   l < 100,
   l++,
   Map[(
      v=#;
      h=RBMSampleHidden[v,wl];
      vprime=RBMSampleVisible[h,wl];
      hprime=RBMSampleHidden[vprime,wl];
      wl[[2]] += .01*
         (Transpose[Transpose[{v}].{h}]
         -Transpose[Transpose[{vprime}].{hprime}]);
      wl[[1,1]] += .01*(v - vprime);
      wl[[1,2]] += .01*(h - hprime);
   )& ,examples]]


RBMTrain[inNet,example]

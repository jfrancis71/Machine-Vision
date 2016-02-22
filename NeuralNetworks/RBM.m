(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


RBMProbHidden[visible_,{biases_,weights_}]:=
   LogisticFn[Flatten[weights.Transpose[{(visible)}]+biases[[2]]]];


RBMSampleHidden[visible_,{biases_,weights_}]:=(
   hidden=Map[Boole[RandomReal[]<#]&,RBMProbHidden[visible,{biases,weights}]]);


RBMProbVisible[hidden_,{biases_,weights_}]:=
   LogisticFn[Flatten[Transpose[weights].Transpose[{(hidden)}]+biases[[1]]]];


RBMSampleVisible[hidden_,{biases_,weights_}]:=
   Map[Boole[RandomReal[]<#]&,RBMProbVisible[hidden,{biases,weights}]]


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
   )& ,examples]];


RBMBuildLayer[from_,to_]:={
{Table[.0,{from}],Table[.0,{to}]},
Table[.0,{to},{from}]};


(* Don't forget to reverse order *)
RBMSampleDownwards[hidden_,{layer_}]:=RBMSampleVisible[hidden,layer];
RBMSampleDownwards[hidden_,layers_List]:=RBMSampleDownwards[RBMSampleVisible[hidden,First[layers]],Rest[layers]];


RBMSampleUpwards[visible_,{layer_}]:=RBMSampleHidden[visible,layer];
RBMSampleUpwards[visible_,layers_List]:=RBMSampleUpwards[RBMSampleHidden[visible,First[layers]],Rest[layers]];


(*RBMTrain[inNet,example]*)


(*
   Ref: https://www.cs.toronto.edu/~hinton/science.pdf
   Hinton and Salakhutdinov, 2006

   Demonstrated good compression.
*)
net={
   RBMBuildLayer[784,1000],
   RBMBuildLayer[1000,500],
   RBMBuildLayer[500,250],
   RBMBuildLayer[250,100],
   RBMBuildLayer[100,30]
};


DBNTrain[net_,examples_]:=(
   dwl=net;
   inp=examples;
   For[lay=1,lay<=Length[net],lay++,
     RBMTrain[dwl[[lay]],inp];dwl[[lay]]=wl;
   inp=Map[RBMSampleHidden[#,wl]&,inp];
   ];
   dwl)


DBNunroll[{rbm_}]:={FullyConnected1DTo1D[rbm[[1,2]],rbm[[2]]],Logistic,FullyConnected1DTo1D[rbm[[1,1]],Transpose[rbm[[2]]]]}


DBNunroll[dbn_List]:=Join[
   {FullyConnected1DTo1D[dbn[[1,1,2]],dbn[[1,2]]],Logistic},
   DBNunroll[Rest[dbn]],
   {FullyConnected1DTo1D[dbn[[1,1,1]],Transpose[dbn[[1,2]]]],Logistic}]

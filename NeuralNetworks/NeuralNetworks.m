(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"


ForwardPropogation[inputs_,{w_,b_}]:=(
A1=Map[Flatten[w].Flatten[#]+b&,inputs];
Z1=ArcTan[A1];
A1
)


randPatches=Table[Random[],{p,1,1000},{y,1,7},{x,1,11}];


initParams={Table[Random[]-0.5,{y,1,7},{x,1,11}],Random[]-0.5};


ForwardPropogation[randPatches,initParams];


Error[inputs_,targets_,params_]:=Total[(ForwardPropogation[inputs,params]-targets)^2]/77


Error[randPatches,targets=Map[Flatten[#].Flatten[leftEye]/100&,randPatches],initParams]


8.338169874338718`


gradMask=Table[ReplacePart[ConstantArray[0,{7,11}],{y,x}->.0001],{y,1,7},{x,1,11}];


Grad[trainingInputs_,trainingTargets_,initialParams_]:=10000*({Map[Error[trainingInputs,trainingTargets,initialParams+{#,0}]&,gradMask,{2}],
Error[trainingInputs,trainingTargets,initialParams+{0,.0001}]}-
Error[trainingInputs,trainingTargets,initialParams])


BatchTraining[trainingInputs_,trainingTargets_,initialParams_,steps_]:=
NestList[#-.00001*Grad[trainingInputs,trainingTargets,#]&,initialParams,steps]


BatchTraining1[trainingInputs_,trainingTargets_,initialParams_,steps_]:=(
p={initialParams};
Print[Dynamic[{f,Error[randPatches,targets,Last[p]],p[[-1,2]]}]];
Table[
   tp = Last[p];
   Table[tp=tp-.00001*Grad[trainingInputs,trainingTargets,tp],{l,1,10}];
   p=Append[p,tp],
{f,1,steps}])


paramsList:=(BatchTraining1[randPatches,targets,Last[p],125000];//AbsoluteTiming)


(*p=Import["c:/users/julian/LastP.wdx"]*)

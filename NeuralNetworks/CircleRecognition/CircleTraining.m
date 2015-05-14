(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"


(*
returns the delta kernel resulting from input mapping onto output
  input: n*s*s
output: n*s*s where s is image size, i.e. 16*16 or 8*8 or 4*4
*)
KernelDeltas[inputs_,outputs_]:=Transpose[Table[Map[Total[Flatten[#]]&,Drop[outputs,0,-j,-i]*Drop[inputs,0,j,i]],{j,-2,+2},{i,-2,+2}],{2,3,1}]


GradE1[w_,labels_,ims_]:=(
images=ToPackedArray[ims];

Timer["ForProp=",
ForwardPropogation[images,w];];

Timer["DeltaO=",DeltaO=
Table[SoftMax[OutputLayer[[n]]][[k]]-Boole[k-1==labels[[n]]],{n,1,Length[labels]},{k,1,2}];];

Timer["gw4=",gw4=
Table[DeltaO[[n,j]] If[i==0,1,H3[[n,i]]],{n,1,Length[labels]},{j,1,2},{i,0,30}];];

Timer["DeltaH3=",DeltaH3=
Table[1/(1+H3A[[n,j]]^2) Sum[DeltaO[[n,k]] w[[4,k,j+1]],{k,1,2}],{n,1,Length[labels]},{j,1,30}];];

Timer["gw3=",gw3 =
Table[DeltaH3[[n,j]] If[i==0,1,Flatten[H2[[n]]][[i]]],
{n,1,Length[labels]},{j,1,30},{i,0,192}];];

(*Timer["DeltaH2=",DeltaH2=
Table[1/(1+Flatten[H2A[[n]]][[j]]^2) Sum[DeltaH3[[n,k]] w[[3,k,j+1]],{k,1,30}],{n,1,Length[images]},{j,1,192}];];
*)

Timer["DeltaH2=",DeltaH2=
Table[1/(1+Flatten[H2A[[n]]][[j]]^2),{n,1,Length[labels]},{j,1,192}] *
DeltaH3.Drop[w[[3]],0,1];];
(*
ptmp=ToPackedArray[Table[Map[Partition[#,4]&,Partition[DeltaH2[[n]],16]],{n,1,Length[labels]}]];
pH1tmp=ToPackedArray[Table[ArrayPad[H1[[n,f2]],2],{n,1,Length[labels]},{f2,1,12}]];
*)
(*
Timer["gw2=",gw2=Table[Prepend[
Flatten[Table[Flatten[ptmp[[n,f1]]]. Extract[pH1tmp[[n]],coords[[f2,cr,cc]]],{cr,1,5},{cc,1,5}]],Total[Partition[DeltaH2[[n]],16][[f1]]]],{n,1,Length[labels]},{f1,1,12},{f2,1,12}];];
*)

Timer["partitioning DeltaH2",partitionedDeltaH2=Map[Map[Partition[#,4]&,Partition[#,16]]&,DeltaH2];];

Timer["upSamplingDeltaH2",upSampleDeltaH2=
ToPackedArray[Table[If[EvenQ[r]&&EvenQ[c],partitionedDeltaH2[[n,f1,r/2,c/2]],0.0],{n,1,Length[labels]},{f1,1,12},{r,1,8},{c,1,8}]];];

Timer["gw2=",gw2=
Transpose[Table[
MapThread[
Prepend,
{Map[Flatten,KernelDeltas[H1[[All,f2]],upSampleDeltaH2[[All,f1]]]],
Map[Total[Partition[#,16][[f1]]]&,DeltaH2]}],
{f1,1,12},{f2,1,12}],{2,3,1}];];

(*ptmp=Table[Map[ArrayPad[Partition[#,4],2]&,Partition[DeltaH2,16]],{n,1,Length[labels]}];*)

(*upSampleDeltaH2=Table[If[EvenQ[r]&&EvenQ[c],Map[Partition[#,4]&,Partition[DeltaH2[[n]],16]][[f1,r/2,c/2]],0.0],{n,1,Length[labels]},{f1,1,12},{r,1,8},{c,1,8}];*)

(*DeltaH1=Table[1/(1+H1A[[f1,r,c]]^2)*
Sum[If[EvenQ[r-j]&&EvenQ[c-i], 
ptmp[[f2,(r-j)/2+2,(c-i)/2+2]],0]*w[[2,f2,f1,(j+2)*5+i+3+1]],{f2,1,12},{j,-2,+2},{i,-2,+2}],{f1,1,12},{r,1,8},{c,1,8}];
Return[];*)

Timer["kernelPts=",kernelPts=
Table[Partition[Rest[w[[2,f2,f1]]],5],{f1,1,12},{f2,1,12}];];

Timer["DeltaH1=",
precomp=Transpose[
Table[
Sum[
ListConvolve[{kernelPts[[f1,f2]]},upSampleDeltaH2[[All,f2]],{1,3,3},0],{f2,1,12}],{f1,1,12}]];
DeltaH1=Table[
Table[1/(1+H1A[[n,f1,r,c]]^2),{r,1,8},{c,1,8}]*
precomp[[n,f1]],{n,1,Length[labels]},{f1,1,12}];];
(*
Timer["upSampleDeltaH1=",
upSampleDeltaH1=ToPackedArray[Table[If[EvenQ[r]&&EvenQ[c],DeltaH1[[n,f1,r/2,c/2]],0.0],{n,1,Length[labels]},{f1,1,12},{r,1,16},{c,1,16}]];];
*)
Timer["upSampleDeltaH1=",upSampleDeltaH1=
ToPackedArray[Table[
Riffle[Map[Riffle[#,0.0,{1,15,2}]&,DeltaH1[[n,f1]]],{ConstantArray[0.0,16]},{1,15,2}],{n,1,Length[labels]},{f1,1,12}]];];

(*Timer["gw1=",gw1=
Table[
Prepend[Flatten[Table[Flatten[Drop[upSampleDeltaH1[[n,f1]],-j,-i]].Flatten[Drop[images[[n]],j,i]],{j,-2,+2},{i,-2,+2}]],
Total[DeltaH1[[n,f1]],2]],
{n,1,Length[labels]},{f1,1,12}]];
*)
Timer["gw1=",gw1=Transpose[Table[
MapThread[
Prepend,
{Map[Flatten,KernelDeltas[images,upSampleDeltaH1[[All,f1]]]],
Map[Total[#,2]&,DeltaH1,{2}][[All,f1]]}],
{f1,1,12}],{2,1,3}];];

Map[Total,{gw1,gw2,gw3,gw4}]
)


BatchGradientDescent[w_]:=w-(0.00001*(wtmp=
Sum[GradE1[w,TrainingLabels[[s*1000+1;;s*1000+1000]],TrainingImages[[s*1000+1;;s*1000+1000]]],{s,0,0}]
))


GradientDescent[w_]:=StochasticGradientDescent[w]


StochasticGradientDescent[w_]:=(
wret=w;
Table[
wret=wret-0.0001*GradE1[wret,TrainingLabels[[s*100+1;;s*100+100]],TrainingImages[[s*100+1;;s*100+100]]],{s,0,9}];
wret )


Length[TrainingImages]


1000


Timer[message_,expr_]:=If[timers==True,Print[message,Timing[expr;]],expr;]


SetAttributes[Timer,HoldAll]


winitial={
Table[Random[]-0.5,{f1,1,12},{i,0,25}],
Table[Random[]-0.5,{f1,1,12},{f2,1,12},{i,0,25}],
Table[Random[]-0.5,{u1,1,30},{in,0,192}],
Table[Random[]-0.5,{u1,1,2},{in,0,30}]
};


myF[w_]:=Function[inputs,Map[SoftMax,ForwardPropogation[inputs,w]]]


Energy[w_]:=CATCrossEntropy[TrainingLabels[[1;;100]]+1, RTrainingImages[[1;;100]], myF[w] ]


StartTraining[freshStart_:True]:=(
Print[Dynamic[{iter,energy[[-5;;-1]]}]];
CATTraining[freshStart];
)

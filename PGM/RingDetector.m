(* ::Package:: *)

(* ::Text:: *)
(*Code assumes a bright background, with ink of .77 brightness. It needs to be seeded with starting point.*)


(* ::Input:: *)
(*rands=Table[{Random[],Random[]},{100}];*)


covar=Table[3 Exp[-2.7 Sin[\[Pi] (x1-x2)]^2],{x1,0,1-(1/6),1/6},{x2,0,1-(1/6),1/6}];


(* ::Input:: *)
(*ShapeFunc[x3_,shape_]=Table[3 Exp[-2.7 Sin[\[Pi] (x3-x1)]^2],{x1,0,1-(1/6),1/6}].Inverse[covar].shape;*)


(* ::Input:: *)
(*DrawRing[patch_,shape_]:=Show[{patch//DispImage,ParametricPlot[{9,9}+(4.5+ShapeFunc[x3/(2 \[Pi]),shape])*{Cos[x3],Sin[x3]},{x3,0,2 \[Pi]}],*)
(*ParametricPlot[{9,9}+(5.5+ShapeFunc[x3/(2 \[Pi]),shape])*{Cos[x3],Sin[x3]},{x3,0,2 \[Pi]}]}]*)


(* ::Input:: *)
(*cache=Table[Table[3 Exp[-2.7` Sin[\[Pi] ((ArcTan[x-9+rands[[n,1]],y-9+rands[[n,2]]]/(2 \[Pi]))-x1)]^2],{x1,0,1-1/6,1/6}].Inverse[covar],{y,1,17},{x,1,17},{n,1,100}];*)


(* ::Input:: *)
(*PercentBlockWithinCircle[y_,x_,shape_]:=*)
(*Sum[Boole[Norm[{y-9-.5,x-9-.5}+rands[[n]]]<(5+cache[[y,x,n]].shape)],{n,1,100}]/100.0*)


(* ::Input:: *)
(*PercentBlockWithinRing[y_,x_,shape_]:=*)
(*PercentBlockWithinCircle[y,x,shape,5+0.5]-PercentBlockWithinCircle[y,x,shape,5-0.5]*)


(* ::Input:: *)
(*RingApp[patch_,shape_]:=*)
(*ShapePrior[shape]+*)
(*200*Sum[-((1*(1-PercentBlockWithinRing[y,x,shape])+0.77*PercentBlockWithinRing[y,x,shape])-patch[[y,x]])^2,{y,1,17},{x,1,17}]*)


GradientShape1[patch_,shape_]:=
{
RingApp[patch,shape+.5*{1,0,0,0,0,0}],
RingApp[patch,shape+.5*{0,1,0,0,0,0}],
RingApp[patch,shape+.5*{0,0,1,0,0,0}],
RingApp[patch,shape+.5*{0,0,0,1,0,0}],
RingApp[patch,shape+.5*{0,0,0,0,1,0}],
RingApp[patch,shape+.5*{0,0,0,0,0,1}]
}-RingApp[patch,shape]


(* ::Input:: *)
(*GradientAscent[patch_]:=NestList[0.3*Normalize[GradientShape1[patch,#]]+#&,baseShape,10];//AbsoluteTiming*)

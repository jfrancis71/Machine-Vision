(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"


surfAngle=Table[If[x==0&&y==0,0.0,
Mod[ArcTan[x,y]+\[Pi]/2,\[Pi]]//N],{y,-8,+8},{x,-8,+8}];


(* This assumes interior is dark and surrounded
   by brighter disk, eg No Entry traffic sign *)
edgeAngle=Table[If[x==0&&y==0,0.0,
ArcTan[x,y]//N],{y,-8,+8},{x,-8,+8}];


circleKernel=Table[If[(5<=Sqrt[(y-9)^2+(x-9)^2]<=7),1.0,0.0],{y,1,17},{x,1,17}];


SurfCircleConv[surfPyramid_]:=
   MVCorrelatePyramid[Cos[2.0*surfPyr], circleKernel*Cos[2.0*surfAngle]] +
      MVCorrelatePyramid[Sin[2.0*surfPyr],circleKernel*Sin[2.0*surfAngle]]


EdgeCircleConv[edgePyramid_]:=
   EdgeDirConv[edgePyr, circleKernel, edgeAngle]


EdgeMagCircleConv[edgeMagPyramid_]:=
   MVCorrelatePyramid[edgeMagPyramid,circleKernel]


EdgeCirclePatch[patch_]:=Cos[patch]*circleKernel*Cos[edgeAngle]+Sin[patch]*circleKernel*Sin[edgeAngle]


Analysis[]:={
   PyramidExplorer[positives[[1]],edgePyr,{ptcht=#;EdgeDirPlot[#],EdgeCirclePatch[#]//Reverse//MatrixPlot}&,{7,146,141}],
   Show[
   circleKernel//Raster//Graphics,
   ptcht//EdgeDirPlot
]}


Assert[surfAngle[[9,17]]==\[Pi]/2];

Assert[edgeAngle[[9,17]]==0];
Assert[edgeAngle[[9,1]]==\[Pi]];
Assert[edgeAngle[[5,5]]==-3 \[Pi]/4];

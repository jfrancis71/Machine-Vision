(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"


surfAngle=Table[If[x==0&&y==0,0.0,
Mod[ArcTan[x,y]+\[Pi]/2,\[Pi]]//N],{y,-8,+8},{x,-8,+8}];


(* This assumes interior is dark and surrounded
   by brighter disk, eg No Entry traffic sign *)
darkInteriorCircleEdgeAngle=Table[If[x==0&&y==0,0.0,
ArcTan[x,y]//N],{y,-8,+8},{x,-8,+8}];


circleKernel=Table[If[(5<=Sqrt[y^2+x^2]<=7),1.0,0.0],{y,-8,+8},{x,-8,+8}];


highResCircleKernel=Table[If[(10<=Sqrt[y^2+x^2]<=11),1.0,0.0],{y,-12,+12},{x,-12,+12}];


SurfCircleToP[x_]:=1/(1+E^-(-20+ .5 *x)); SetAttributes[SurfCircleToP,Listable];
SurfCircleConv[surfPyramid_]:=
   (MVCorrelatePyramid[Cos[2.0*surfPyramid], circleKernel*Cos[2.0*surfAngle]] +
      MVCorrelatePyramid[Sin[2.0*surfPyramid],circleKernel*Sin[2.0*surfAngle]])//SurfCircleToP


EdgeCircleConv[edgePyramid_]:=
   EdgeDirConv[edgePyramid, circleKernel, darkInteriorCircleEdgeAngle]


EdgeMagCircleConv[edgeMagPyramid_]:=
   MVCorrelatePyramid[edgeMagPyramid,circleKernel]


EdgeCirclePatchH[patch_]:=Cos[patch]*circleKernel*Cos[edgeAngle]+Sin[patch]*circleKernel*Sin[edgeAngle];
EdgeCirclePatch[patch_]:=Total[EdgeCirclePatchH[patch],2]


SurfCirclePatchH[patch_]:=0.5*Cos[2.0*(patch-surfAngle)]*circleKernel;
SurfCirclePatch[patch_]:=1.0/(1.0+E^(-Total[SurfCirclePatchH[patch],2]+20.0));


Analysis[]:={
   PyramidExplorer[positives[[1]],edgePyr,{ptcht=#;EdgeDirPlot[#],EdgeCirclePatch[#]//Reverse//MatrixPlot}&,{7,146,141}],
   Show[
   circleKernel//Raster//Graphics,
   ptcht//EdgeDirPlot
]}


HNDist[x_,s_]=FullSimplify[PDF[HalfNormalDistribution[s],x],Assumptions->{x>0}]


BesI0s5=1/(\[Pi] BesselI[0,5])//N


CircleSurfMagDirPatchH[patch_]:=
Log[2,((* This is the e = 0 case *)
(0.57*HNDist[patch[[All,All,1]],11.1]+(1.0-.57)*HNDist[patch[[All,All,1]],1.6]) (*p[Edge Mag|E=0]*)
*
1/\[Pi](*p[edge dir|E=0*)
*(1-circleKernel)(*p[E=0]*)
)+
(* This is the e = 1 case *)
HNDist[patch[[All,All,1]],1.6](*p[Edge Mag|E=1*)*
(0.57*BesI0s5*E^(05.0 Cos[2.0*(patch[[All,All,2]]-surfAngle)])+(1.0-0.57)*1/\[Pi]) * (*p[Edge dir|E=1*)
circleKernel (*p[E=1]*)]-
Log[2,1/\[Pi]*(0.57*HNDist[patch[[All,All,1]],11.1]+(1-.57)*HNDist[patch[[All,All,1]],1.6])]


CircleSurfMagDirPatch[patch_]:=Total[CircleSurfMagDirPatchH[patch],2]


On[Assert]

Assert[surfAngle[[9,17]]==\[Pi]/2];

Assert[darkInteriorCircleEdgeAngle[[9,17]]==0];
Assert[darkInteriorCircleEdgeAngle[[9,1]]==\[Pi]];
Assert[darkInteriorCircleEdgeAngle[[5,5]]==-3 \[Pi]/4];

(
   testImage=StandardiseImage["c:/users/julian/my documents/github/Machine-Vision/coin3.jpg"];
   testSurfPyr=SobelFilter[BuildPyramid[testImage,{8,8}],SurfDir];
   testSurfConv=SurfCircleConv[testSurfPyr];
   Assert[ 0.95 < testSurfConv[[1,39,76]] < 1.01 ];
   testFiltOut=SurfCirclePatch[testSurfPyr[[1,39-8;;39+8,76-8;;76+8]]];
   Assert[ 0.95 < testFiltOut < 1.01];
   Assert[ Abs[testSurfConv[[1,39,76]] - testFiltOut ] < 0.01 ];
)

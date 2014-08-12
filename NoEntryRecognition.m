(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/CircleDetectors.m"


positiveFiles=FileNames["C:\\Users\\Julian\\secure\\Shape Recognition\\Stop Sign\\Training Data\\Positives\\*"];


positives=Map[StandardiseImage[#,640]&,positiveFiles];


negativeFiles=FileNames["C:\\Users\\Julian\\secure\\Shape Recognition\\Stop Sign\\Training Data\\Negatives\\*.JPG"];


negatives=Map[StandardiseImage[#,640]&,negativeFiles];


barAngle=Table[
   If[Abs[x]<6,If[y==1,-\[Pi]/2,If[y==-1,\[Pi]/2,0]],0]
,{y,-8,8},{x,-8,8}];


barKernel=Table[
   If[Abs[x]<6,If[y==1||y==-1,1,0],0]
,{y,-8,+8},{x,-8,+8}];


BrightBarSurfMagDirPatchH[patch_]:=
   Log[2,HNDist[patch[[All,All,1]],1.6](*p[Edge Mag|E=1*)*
   BesI0s5*E^(05.0 Cos[patch[[All,All,2]]-barAngle]) * (*p[Edge dir|E=1*)
   barKernel (*p[E=1]*)] -
   Log[2,HNDist[patch[[All,All,1]],11.1]*1/(2 \[Pi])]*barKernel;
BrightBarSurfMagDirPatch[patch_]:=BrightBarSurfMagDirPatch[patch]


mylog2[x_]=N[1.0/(1.0+Exp[-(-800.0+ x)]),5];


SetAttributes[mylog2,Listable];
SetAttributes[mylog2,NumericFunction];


BrightBarPatchH[patch_]:=Cos[(patch-barAngle)]*barKernel;
BrightBarPatch[patch_]:=Total[BrightBarPatchH[patch],2]//mylog2


NoEntryRecognition[image_?MatrixQ]:= (
   imgPyramid=BuildPyramid[image,{8,8}];
Print["Build Pyr"];
   edgeMagDir=SobelFilter[imgPyramid,EdgeMagDir];
   surfPyr=EdgeMagDirToSurfDir[edgeMagDir];
Print["Build Surf Pyr"];
   edgeDirPyr=EdgeMagDirToEdgeDir[edgeMagDir];
Print["Build Edge Dir Pyr"];
   edgeMagPyr=EdgeMagDirToEdgeMag[edgeMagDir];
Print["Build Edge Mag Pyr"];
   circleFilterOutput=SurfCircleConv[surfPyr];
Print["Build Circle Pyr"];
(*
   barOutput=
      MVCorrelatePyramid[Cos[2.0*surfPyr], barKernel*Cos[2.0*barAngle]]+
      MVCorrelatePyramid[Sin[2.0*surfPyr], barKernel*Sin[2.0*barAngle]];
  
*)
   barOutput=EdgeDirConv[edgeDirPyr,barKernel,barAngle]
     +MVCorrelatePyramid[Log[2,HNDist[edgeMagPyr,1.6]],barKernel]
     -MVCorrelatePyramid[Log[2,0.57*HNDist[edgeMagPyr,11.2]+(1.0-0.57)*HNDist[edgeMagPyr,1.6]],barKernel];
Print["Build bar"];
(*   circleFilterOutput*mylog2[barOutput];*)

   pl1=circleFilterOutput;
Print["Build pl1"];
   pr1=1.0/(1.0+Exp[-(-40.0+barOutput)]); (*Unfortunate can't get mylog to work machine precision *)
Print["Build mylog2"];
   pl2 = 0.02*pl1 + 10^-7*(1-pl1);
   pr2 = 0.0005*pr1 + 10^-7*(1-pr1);

   pf=10^-6;
  res=(pf*(1-pf))/(pl2*pr2*(1-pf) + (1-pl2)*(1-pr2)*pf)*(pl2*pr2)/pf;
Print["Completed"];
res
)


NoEntryRecognitionOutput[image_?MatrixQ,threshold_:.2]:= (
   output=NoEntryRecognition[image];
   Show[image//DispImage,BoundingRectangles[output,threshold,{8,8}]//OutlineGraphics] )

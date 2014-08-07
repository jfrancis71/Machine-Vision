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


mylog2[x_]:=1/(1+E^-(-20+ x))


SetAttributes[mylog2,Listable]


NoEntryRecognition[image_?MatrixQ]:= (
   imgPyramid=BuildPyramid[image,{8,8}];
   surfPyr=SobelFilter[imgPyramid,SurfDir];
   edgeDirPyr=SobelFilter[imgPyramid,EdgeDir];
   circleFilterOutput=SurfCircleConv[surfPyr];
(*
   barOutput=
      MVCorrelatePyramid[Cos[2.0*surfPyr], barKernel*Cos[2.0*barAngle]]+
      MVCorrelatePyramid[Sin[2.0*surfPyr], barKernel*Sin[2.0*barAngle]];
  
*)
   barOutput=EdgeDirConv[edgeDirPyr,barKernel,barAngle];

   circleFilterOutput*mylog2[barOutput];

   pl1=mylog2[barOutput];
   pr1=circleFilterOutput'

   pl2 = 0.02*pl1 + 10^-7*(1-pl1);
   pr2 = 0.0002*pr1 + 10^-7*(1-pr1);

   pf=10^-6;
  (pf*(1-pf))/(pl2*pr2*(1-pf) + (1-pl2)*(1-pr2)*pf)*(pl2*pr2)/pf]]

)


NoEntryRecognitionOutput[image_?MatrixQ,threshold_:.2]:= (
   output=NoEntryRecognition[image];
   Show[image//DispImage,BoundingRectangles[output,threshold,{8,8}]//OutlineGraphics] )

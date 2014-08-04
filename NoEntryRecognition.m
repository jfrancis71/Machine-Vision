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


mylog1[x_]:=1/(1+E^-(-20+ .5 *x))


mylog2[x_]:=1/(1+E^-(-20+ x))


SetAttributes[mylog1,Listable];SetAttributes[mylog2,Listable]


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

   mylog1[circleFilterOutput]*mylog2[barOutput]
)


NoEntryRecognition1[image_?MatrixQ]:= (
   imgPyramid=BuildPyramid[image,{8,8}];
   edgeMagPyr=SobelFilter[imgPyramid,EdgeMag];
   circleFilterOutput=EdgeMagCircleConv[edgeMagPyr]
)


NoEntryRecognitionOutput[image_?MatrixQ,maxNo_:100]:= (
   output=NoEntryRecognition[image];
   threshold = RankedMax[Flatten[output],maxNo];
   Show[image//DispImage,BoundingRectangles[output,threshold,{8,8}]//OutlineGraphics] )

(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"


positiveFiles=FileNames["C:\\Users\\Julian\\secure\\Shape Recognition\\Stop Sign\\Training Data\\Positives\\*"];


positives=Map[StandardiseImage[#,640]&,positiveFiles];


negativeFiles=FileNames["C:\\Users\\Julian\\secure\\Shape Recognition\\Stop Sign\\Training Data\\Negatives\\*.JPG"];


negatives=Map[StandardiseImage[#,640]&,negativeFiles];


circleAngle=Table[If[x==0&&y==0,0.0,
Mod[ArcTan[x,y]+\[Pi]/2,\[Pi]]//N],{y,-8,+8},{x,-8,+8}];


circleKernel=Table[If[(5<=Sqrt[(y-9)^2+(x-9)^2]<=7),1.0,0.0],{y,1,17},{x,1,17}];


NoEntryRecognition[image_?MatrixQ]:= (
   imgPyramid=BuildPyramid[image,{8,8}];
   edgePyr=SobelFilter[imgPyramid,SurfDir];
   circleFilterOutput=MVCorrelatePyramid[Cos[2.0*edgePyr], circleKernel*Cos[2.0*circleAngle]] +
      MVCorrelatePyramid[Sin[2.0*edgePyr],circleKernel*Sin[2.0*circleAngle]]
)


NoEntryRecognitionOutput[image_?MatrixQ]:= (
   output=NoEntryRecognition[image];
   threshold = RankedMax[Flatten[output],100];
   Show[image//DispImage,BoundingRectangles[output,threshold,{8,8}]//OutlineGraphics] )

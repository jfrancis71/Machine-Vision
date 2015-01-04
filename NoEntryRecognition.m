(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/CircleDetectors.m"


positiveFiles=FileNames["C:\\Users\\Julian\\secure\\Shape Recognition\\Stop Sign\\Training Data\\Positives\\*"];


positives=Map[StandardiseImage[#,640]&,positiveFiles];


negativeFiles=FileNames["C:\\Users\\Julian\\secure\\Shape Recognition\\Stop Sign\\Training Data\\Negatives\\*.JPG"];


negatives=Map[StandardiseImage[#,640]&,negativeFiles];


NoEntrySignMask=Table[
   If[Sqrt[x^2+y^2]<=6.0,1.0,0.0]
      ,{y,-8,+8},{x,-8,+8}];

NoEntryEdgeKernel=Table[If[(7<Sqrt[y^2+x^2]<=8),1.0,0.0],{y,-8,+8},{x,-8,+8}];
NoEntrySignExt=Position[NoEntryEdgeKernel,1.];
NoEntrySignInt=Map[Function[c,SortBy[Position[NoEntrySignMask,1.],EuclideanDistance[#,c]&]//First],Position[NoEntryEdgeKernel,1.]];

MVShapeNoEntryFilt[patch_]:=Total[Clip[(Extract[patch,NoEntrySignExt]-Extract[patch,NoEntrySignInt])^2,{0,0.2^2}]]


NoEntrySignKernel=Table[
   If[Sqrt[x^2+y^2]<=6.0,If[Abs[x]<=5&&Abs[y]<=1,.8,.8*.4],0.0]
      ,{y,-8,+8},{x,-8,+8}];


AppearanceNoEntryConvFast[pyr_]:=(
   S1=MVCorrelatePyramid[pyr^2,
   NoEntrySignMask];
   S2=MVCorrelatePyramid[pyr,NoEntrySignKernel];
   S3=Total[NoEntrySignKernel^2,2];
   (-1+Log[Exp[-S1] Exp[S2^2/S3]* ( Erf[(S3-S2)/Sqrt[S3]] + Erf[(S2-0.25 S3)/Sqrt[S3]] )])
)


erasePyr=Map[ConstantArray[1.,Dimensions[#]]&,BuildPyramid[positives[[1]],{8,8}][[1;;-1]]];
Table[erasePyr[[l,1;;8,All]]=0.,{l,1,Length[erasePyr]}];
Table[erasePyr[[l,-8;;-1,All]]=0.,{l,1,Length[erasePyr]}];
Table[erasePyr[[l,All,1;;8]]=0.,{l,1,Length[erasePyr]}];
Table[erasePyr[[l,All,-8;;-1]]=0.,{l,1,Length[erasePyr]}];


NoEntryRecognition[image_?MatrixQ]:=(
   pyr=BuildPyramid[image,{8,8}];

   rawShape=PyramidFilter[MVShapeNoEntryFilt, pyr, {8,8}];
   shape=1/(1+Exp[-(rawShape-.58)*50]);
   shape = shape*erasePyr;

   rawApp=AppearanceNoEntryConvFast[pyr];
 (* 120 *)
   appearance=1/(1+Exp[-(rawApp+0.9)*50]);
   appearance=appearance*erasePyr;

   res=appearance*shape
)


NoEntryRecognitionOutput[image_?MatrixQ,threshold_:.5]:= (
   output=NoEntryRecognition[image];
   Show[image//DispImage,BoundingRectangles[output,threshold,{8,8}]//OutlineGraphics] )

(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"


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


AngleDiff[a_,b_]:=Min[Abs[a-b],2 \[Pi] - Abs[a-b]]


NoEntryBoundaryKernel=Table[If[(6<Sqrt[y^2+x^2]<=7),1.0,0.0],{y,-8,+8},{x,-8,+8}];


NoEntryEdgeVKernel=Table[If[x==0&&y==0,0,If[
   AngleDiff[\[Pi]/2,ArcTan[x,y]]<((\[Pi]/2)/3)||AngleDiff[-\[Pi]/2,ArcTan[x,y]]<((\[Pi]/2)/3)
      ,1.,0.]],{y,-8,+8},{x,-8,+8}]*NoEntryBoundaryKernel;


NoEntryEdgeHKernel=Table[If[x==0&&y==0,0,If[
   AngleDiff[\[Pi],ArcTan[x,y]]<((\[Pi]/2)/3)||AngleDiff[0,ArcTan[x,y]]<((\[Pi]/2)/3)
      ,1.,0.]],{y,-8,+8},{x,-8,+8}]*NoEntryBoundaryKernel;


NoEntryEdgeDiag1Kernel=Table[If[x==0&&y==0,0,If[
   AngleDiff[\[Pi]/4,ArcTan[x,y]]<((\[Pi]/2)/3)-.2||AngleDiff[-3 \[Pi]/4,ArcTan[x,y]]<((\[Pi]/2)/3)-.2
      ,1.,0.]],{y,-8,+8},{x,-8,+8}]*NoEntryBoundaryKernel;


NoEntryEdgeDiag2Kernel=Table[If[x==0&&y==0,0,If[
   AngleDiff[3 \[Pi]/4,ArcTan[x,y]]<((\[Pi]/2)/3)-.2||AngleDiff[- \[Pi]/4,ArcTan[x,y]]<((\[Pi]/2)/3)-.2
      ,1.,0.]],{y,-8,+8},{x,-8,+8}]*NoEntryBoundaryKernel;


ShapeNoEntryPatch[patch_]:=-(
Log[
   0.3*(15.9/(1.+(2500.*patch[[1]] ))) + 0.7
      ]*NoEntryEdgeHKernel+
Log[
   0.3*(15.9/(1.+(2500.*patch[[2]] ))) + 0.7
      ]*NoEntryEdgeVKernel+
Log[
   0.3*(15.9/(1.+(2500.*patch[[3]] ))) + 0.7
      ]*NoEntryEdgeDiag1Kernel+
Log[
   0.3*(15.9/(1.+(2500.*patch[[4]] ))) + 0.7
      ]*NoEntryEdgeDiag2Kernel
)


(* ::Text:: *)
(*PDF[StudentTDistribution[0,.02,1],d]  =  15.9155/(1+2500. d^2)*)


ShapeNoEntryConv[pyr_]:=(
   horizDiff=MVCorrelatePyramid[pyr,{{1,0,-1}}]^2;
   vertDiff=MVCorrelatePyramid[pyr,Transpose[{{1,0,-1}}]]^2;
   diag1Diff=MVCorrelatePyramid[pyr,{{0,0,1},{0,0,0},{-1,0,0}}//Reverse]^2;
   diag2Diff=MVCorrelatePyramid[pyr,{{0,0,1},{0,0,0},{-1,0,0}}]^2;
   -(
    MVCorrelatePyramid[Log[0.3*15.9/(1+horizDiff*2500)+0.7],NoEntryEdgeHKernel]+
    MVCorrelatePyramid[Log[0.3*15.9/(1+vertDiff*2500)+0.7],NoEntryEdgeVKernel]+
    MVCorrelatePyramid[Log[0.3*15.9/(1+diag1Diff*2500)+0.7],NoEntryEdgeDiag1Kernel]+
    MVCorrelatePyramid[Log[0.3*15.9/(1+diag2Diff*2500)+0.7],NoEntryEdgeDiag2Kernel]
)
)


NoEntrySignKernel=Table[
   If[Sqrt[x^2+y^2]<=6.0,If[Abs[x]<=5&&Abs[y]<=1,.8,.8*.4],0.0]
      ,{y,-8,+8},{x,-8,+8}];


AppearanceNoEntryConvFast[pyr_]:=(
   S1=MVCorrelatePyramid[pyr^2,
   NoEntrySignMask];
   S2=MVCorrelatePyramid[pyr,NoEntrySignKernel];
   S3=Total[NoEntrySignKernel^2,2];
   (-1+Log[Exp[-S1] Exp[S2^2/S3]* ( Erf[(S3-S2)/Sqrt[S3]] + Erf[(S2-0.25 S3)/Sqrt[S3]] )]);

   S1 = (1./(2. .0025)) * MVCorrelatePyramid[pyr^2,NoEntrySignMask];
   S2 = (1./(2. .0025)) * MVCorrelatePyramid[pyr,NoEntrySignKernel];
   S3 = (1./(2. .0025)) * Total[NoEntrySignKernel^2,2];

   P1 = ( Erf[(S3-S2)/Sqrt[S3]] + Erf[(S2 - 0.25 S3)/Sqrt[S3]] );
   P2 = Exp[-S1] Exp[S2^2/S3];
   (*Log[1./(Sqrt[2 3.14] .05)^81 .886 * Exp[-S1] Exp[S2^2/S3] ( Erf[(S3-S2)/Sqrt[S3]] + Erf[S2/Sqrt[S3]] )]- 81 Log[3]*)
   (*Log[1./(Sqrt[2 3.14] .05)^81] - 81 Log[3] + Log[.886] + -S1 + S2^2/S3 + 0.7*)
   Log[(1./(Sqrt[2 3.14] .05)^113) 1.77245 * P2 P1 ]- Log[2 Sqrt[S3]] - 113 Log[3]
)


AppearanceNoEntryPatchHelp[patch_]:=(
Log[Sum[1/((1-.25)/.01) Exp[(-1/(2. .0025)) Sum[NoEntrySignMask[[y,x]] (patch[[y,x]] - \[Alpha] NoEntrySignKernel[[y,x]])^2,{y,1,17},{x,1,17}]],{\[Alpha],.25,1,.01}]]
)

AppearanceNoEntryPatch[patch_]:=Log[10^-20 + 1./(Sqrt[2 3.14] .05)^113 .886 ] + AppearanceNoEntryPatchHelp[patch] - 113 Log[3]


erasePyr=Map[ConstantArray[1.,Dimensions[#]]&,BuildPyramid[positives[[1]],{8,8}][[1;;-1]]];
Table[erasePyr[[l,1;;8,All]]=0.,{l,1,Length[erasePyr]}];
Table[erasePyr[[l,-8;;-1,All]]=0.,{l,1,Length[erasePyr]}];
Table[erasePyr[[l,All,1;;8]]=0.,{l,1,Length[erasePyr]}];
Table[erasePyr[[l,All,-8;;-1]]=0.,{l,1,Length[erasePyr]}];


NoEntryRecognition[image_?MatrixQ]:=(
   pyr=BuildPyramid[image,{8,8}];

   shape=ShapeNoEntryConv[pyr];

   appearance=AppearanceNoEntryConvFast[pyr];
 (* 120 *)
   
   res=appearance+shape
)


NoEntryRecognitionOutput[image_?MatrixQ,threshold_:.5]:= (
   output=NoEntryRecognition[image];
   Show[image//DispImage,BoundingRectangles[output,threshold,{8,8}]//OutlineGraphics] )


GroundTruth=Table[{},{10}];


GroundTruth[[1]]={
{6,162,156},
{2,239,370}};


GroundTruth[[2]]={
{10,120,119},
{4,134,231}};


GroundTruth[[3]]={
{8,107,250}
};


GroundTruth[[4]]={
{8,134,86},
{9,128,220}
};


GroundTruth[[5]]={
{7,138,78},
{9,116,243}
};


GroundTruth[[6]]={
{9,110,218}
};


GroundTruth[[7]]={
{5,157,285}
};


GroundTruth[[8]]={
{6,159,86},
{8,129,247}
};


GroundTruth[[10]]={
{5,160,225}
};

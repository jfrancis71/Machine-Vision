(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"
<<"C:/users/julian/documents/github/Machine-Vision/FeatureDetectors.m"


positiveFiles=FileNames["C:\\Users\\Julian\\secure\\Shape Recognition\\Stop Sign\\Training Data\\Positives\\*"];


positives=Map[StandardiseImage[#,640]&,positiveFiles];


negativeFiles=FileNames["C:\\Users\\Julian\\secure\\Shape Recognition\\Stop Sign\\Training Data\\Negatives\\*.JPG"];


negatives=Map[StandardiseImage[#,640]&,negativeFiles];


ImagesQ[images_]:=MatrixQ[images[[1]]];


CentreMask=ArrayPad[ConstantArray[1,{3,13}],{{2,2},{0,0}}];


CentreKernel=CentreMask;


ExtBarMask=ArrayPad[ConstantArray[0,{3,13}],{{2,2},{1,1}},1];


ExtBarKernel=ExtBarMask;


ExtBarMask=ArrayPad[ConstantArray[0,{5,13}],{{1,1},{0,0}},1];


ExtBarKernel=ExtBarMask;


altMask=Clip[CentreMask+ExtBarMask,{0,1}];


altKernel=altMask;


maskK={ConstantArray[0,21]};maskK[[1,1]]=1;maskK[[1,7]]=1;edgeK=maskK;edgeK[[1,7]]=-1;


myMax[a_,b_]:=Max[a,b]


SetAttributes[myMax,Listable]


NoEntryRecognition[images_?ImagesQ]:=(
   pyrs=Map[BuildPyramid,positives[[1;;9]]];
   \[Lambda]=Map[MVCorrelatePyramid[#,{{1,1,1}}]/3.&,pyrs];
   \[Nu]=Map[MVCorrelatePyramid[#,{{0,0,0},{0,0,0},{0,0,0},{1,1,1}}]/3.&,pyrs];
   centre=Table[TemplateDetector[pyrs[[w]],CentreKernel,CentreMask,\[Lambda][[w]]],{w,1,Length[images]}];
   ext1=Table[TemplateDetector[pyrs[[w]],ExtBarKernel,ExtBarMask,\[Lambda][[w]]*.33],{w,1,Length[images]}];
   ext2=Table[TemplateDetector[pyrs[[w]],ExtBarKernel,ExtBarMask,\[Lambda][[w]]*.52],{w,1,Length[images]}];
   ext3=Table[TemplateDetector[pyrs[[w]],ExtBarKernel,ExtBarMask,\[Lambda][[w]]*.41],{w,1,Length[images]}];
   extBar=myMax[ext1,myMax[ext2,ext3]];
   alt=myMax[
   ct1=Table[TemplateDetector[pyrs[[w]],altKernel,altMask,\[Lambda][[w]]],{w,1,Length[images]}],
   ct2=Table[TemplateDetector[pyrs[[w]],altKernel*.1,altMask],{w,1,Length[images]}]
      ];
   edgeBarL=Table[MVCorrelatePyramid[pyrs[[w]],edgeK],{w,1,Length[images]}];lEdgeBar=2*Log[NeighbourProb[0,edgeBarL]];
   edgeBarR=Table[MVCorrelatePyramid[pyrs[[w]],Map[Reverse,edgeK]],{w,1,Length[images]}];rEdgeBar=2*Log[NeighbourProb[0,edgeBarR]];
   app=centre+extBar;
   alt2=myMax[alt,app+Log[.00001]];
   tot=(Log[.001]+app)-Map[Max[#,115]&,(alt2 + Log[.999]),{4}]-lEdgeBar-rEdgeBar;
   tot=tot*UnitStep[\[Lambda]-.25])



   disp[p_]:=Show[positives[[p]]//DispImage,BoundingRectangles[tot[[p]],4.,{10,10}]//OutlineGraphics] 


NoEntryRecognitionOutput[images_?ImagesQ]:=(
   NoEntryRecognition[images];
   Table[disp[p],{p,1,9}])


NoEntryRecognition[image_?MatrixQ]:=(
   NoEntryRecognition[{image}]
);


NoEntryRecognitionOutput[image_?MatrixQ,threshold_:4.]:= (
   output=NoEntryRecognition[{image}];
   Show[image//DispImage,BoundingRectangles[output,threshold,{8,8}]//OutlineGraphics] );


DisplayPatches[]:=(
   conc=Select[Position[tot,x_/;x>4.0],InBoundsQ[pyrs[[1]],#[[2]],#[[3]],#[[4]],10]&];
   SortBy[Map[
   {#,
   tot[[#[[1]],#[[2]],#[[3]],#[[4]]]],
   app[[#[[1]],#[[2]],#[[3]],#[[4]]]],
   alt2[[#[[1]],#[[2]],#[[3]],#[[4]]]],
   (Patch[pyrs[[#[[1]]]],#[[2]],#[[3]],#[[4]],10]//DispImage)}&,conc],#[[2]]&]);


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

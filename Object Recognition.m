(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"
<<"C:/users/julian/documents/github/Machine-Vision/ObjectModels.m"


ZeroBoundary[pyr_]:=Table[
   If[y<=8||y>Length[pyr[[l]]]-8||
     x<=8||x>Length[pyr[[l,1]]]-8,0.,1.]
   ,{l,1,Length[pyr]},{y,1,Length[pyr[[l]]]},{x,1,Length[pyr[[l,1]]]}]


c1=Table[ArrayPad[ConstantArray[1.,(pyr[[l]]//Dimensions)-{16,16}],8],{l,1,Length[pyr]}];//AbsoluteTiming


probF[z_]=FullSimplify[PDF[HalfNormalDistribution[15],z],Assumptions->z>0];


ObjectRecognition[image_,featureKernels_,partRelationKernels_]:=( 
   pyr=BuildPyramid[image];
   featureMaps=probF[Map[
   MVCorrelatePyramid[pyr,#,NormalizedSquaredEuclideanDistance]&,
     featureKernels]];
   res=Apply[Times,MapThread[MVCorrelatePyramid,{featureMaps,partRelationKernels}]]
)


ObjectRecognitionOutput[image_,featureKernels_,partRelationKernels_]:=
(
   Assert[Length[featureKernels]==Length[partRelationKernels]];
   ObjectRecognition[image,featureKernels,partRelationKernels];
   c1=Table[ArrayPad[ConstantArray[1.,(pyr[[l]]//Dimensions)-{16,16}],8],{l,1,Length[pyr]}];
   o=Show[image//DispImage,
      OutlineGraphics[BoundingRectangles[res,1.0,{10,10}],Green]]
)


ObjectRecognitionOutput[image_]:=
(
   c1=Table[ArrayPad[ConstantArray[1.,(pyr[[l]]//Dimensions)-{16,16}],8],{l,1,Length[pyr]}];
   AlienMap=ObjectRecognition[image,AlienFeatureKernels,AlienRelationKernels];
   CokeMap=ObjectRecognition[image,CokeFeatureKernels,CokeRelationKernels];
   EngSparkMap=ObjectRecognition[image,EngSparkFeatureKernels,EngSparkRelationKernels];
   FaceMap=ObjectRecognition[image,FaceFeatureKernels,FaceRelationKernels];

   o=Show[image//DispImage,
      OutlineGraphics[BoundingRectangles[AlienMap,1.0,{10,10}],Red],
      OutlineGraphics[BoundingRectangles[CokeMap,1.0,{10,10}],Green],
      OutlineGraphics[BoundingRectangles[EngSparkMap,1.0,{10,10}],Blue],
      OutlineGraphics[BoundingRectangles[FaceMap,1.0,{10,10}],Yellow],
      PlotRange->{{0,(image//Dimensions)[[2]]},{0,(image//Dimensions)[[1]]}}
] 
)


FaceRecognitionOutput[image_]:=
(
   c1=Table[ArrayPad[ConstantArray[1.,(pyr[[l]]//Dimensions)-{16,16}],8],{l,1,Length[pyr]}];
   FaceMap=ObjectRecognition[image,FaceFeatureKernels,FaceRelationKernels];

   o=Show[image//DispImage,
      OutlineGraphics[BoundingRectangles[FaceMap,1.0,{10,10}],Yellow],
      PlotRange->{{0,(image//Dimensions)[[2]]},{0,(image//Dimensions)[[1]]}}
] 
)

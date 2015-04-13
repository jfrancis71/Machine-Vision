(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/AlienRecognition.m"


<<"C:/users/julian/documents/github/Machine-Vision/CokeDetector.m"


<<"C:/users/julian/documents/github/Machine-Vision/EngSparkRecognition.m"


ZeroBoundary[pyr_]:=Table[
   If[y<=8||y>Length[pyr[[l]]]-8||
     x<=8||x>Length[pyr[[l,1]]]-8,0.,1.]
   ,{l,1,Length[pyr]},{y,1,Length[pyr[[l]]]},{x,1,Length[pyr[[l,1]]]}]


c1=Table[ArrayPad[ConstantArray[1.,(pyr[[l]]//Dimensions)-{16,16}],8],{l,1,Length[pyr]}];//AbsoluteTiming


ObjectRecognitionOutput[image_]:=
(
   c1=Table[ArrayPad[ConstantArray[1.,(pyr[[l]]//Dimensions)-{16,16}],8],{l,1,Length[pyr]}];
   AlienRecognition[image];AlienMap=res*c1;
   CokeRecognition[image];CokeMap=res*c1;
   EngSparkRecognition[image];EngSparkMap=res*c1;
   o=Show[image//DispImage,
      OutlineGraphics[BoundingRectangles[AlienMap,1.0,{10,10}],Red],
      OutlineGraphics[BoundingRectangles[CokeMap,1.0,{10,10}],Green],
      OutlineGraphics[BoundingRectangles[EngSparkMap,1.0,{10,10}],Blue]
] 
)

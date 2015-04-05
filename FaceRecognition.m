(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"


ModelFace=StandardiseImage[Import["c:/users/julian/secure/Shape Recognition/huttenlocher/images/faces/Training/image_0001.jpg"]
];


leftEye=ImageData[ImageTake[ModelFace//Image,{85,85}-{45,39},{35,45}]];
rightEye=ImageData[ImageTake[ModelFace//Image,{85,85}-{45,39},{53,63}]];


leftEyeKernel=Table[0,{1},{21}];leftEyeKernel[[1,1]]=1;leftEyeKernel[[1,2]]=1;leftEyeKernel[[1,3]]=1;
rightEyeKernel=Table[0,{1},{21}];rightEyeKernel[[1,19]]=1;rightEyeKernel[[1,20]]=1;rightEyeKernel[[1,21]]=1;


probF[z_]=Simplify[PDF[HalfNormalDistribution[15],z],Assumptions->z>0];


FaceRecognition[image_]:=(
   pyr=BuildPyramid[image];
   featureMaps=probF[Map[
   MVCorrelatePyramid[pyr,#,NormalizedSquaredEuclideanDistance]&,
     {leftEye,rightEye}]];
   res=(MVCorrelatePyramid[featureMaps[[1]],leftEyeKernel]*
   MVCorrelatePyramid[featureMaps[[2]],rightEyeKernel]);
)


FaceRecognitionOutput[image_]:=(
   FaceRecognition[image];
   Show[image//DispImage,BoundingRectangles[res,1.0,{10,10}]//OutlineGraphics] );

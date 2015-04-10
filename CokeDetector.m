(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"


cokeModel=StandardiseImage[Import["c:/users/julian/my documents/github/machine-vision/CokeModel.jpg"]];


feature1=cokeModel[[14-5;;14+5,74-5;;74+5]];


feature2=cokeModel[[37-5;;37+5,74-5;;74+5]];


Feature1Kernel=ConstantArray[1,{3,3}];
Feature2Kernel=Table[If[y<=-22,1,0],{y,-25,+25},{x,-2,+2}];


probF[z_]=FullSimplify[PDF[HalfNormalDistribution[15],z],Assumptions->z>0];


CokeRecognition[image_]:=( 
pyr=BuildPyramid[image];
featureMaps=probF[Map[
   MVCorrelatePyramid[pyr,#,NormalizedSquaredEuclideanDistance]&,
     {feature1,feature2}]];
res=(MVCorrelatePyramid[featureMaps[[1]],Feature1Kernel]*
   MVCorrelatePyramid[featureMaps[[2]],Feature2Kernel]);
)


CokeRecognitionOutput[image_]:=( 
CokeRecognition[image];
Show[image//DispImage,BoundingRectangles[res,1.0,{10,10}]//OutlineGraphics] );

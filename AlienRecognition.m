(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"


AlienModel=StandardiseImage[Import["c:/users/julian/my documents/github/machine-vision/AlienModel.jpg"]];


(* Note, don't forget the step 2! *)
AlienFeature1=AlienModel[[62-8;;62+8;;2,62-8;;62+8;;2]];
AlienFeature2=AlienModel[[55-8;;55+8;;2,107-8;;107+8;;2]];
AlienFeature3=AlienModel[[34-5;;34+5,51-4;;51+4;;2]];
Map[DispImage,{AlienFeature1,AlienFeature2,AlienFeature3}]


AlienFeature1Kernel=ConstantArray[1,{3,3}];
AlienFeature2Kernel=Table[If[y<=-4&&x>=21,1,0],{y,-5,+5},{x,-22,+22}];
AlienFeature3Kernel=Table[If[y<-13&&x<-5,1,0],{y,-18,+18},{x,-7,+7}];


probF[z_]=FullSimplify[PDF[HalfNormalDistribution[15],z],Assumptions->z>0];


AlienRecognition[image_]:=( 
   pyr=BuildPyramid[image];
   AlienFeatureMaps=probF[Map[
   MVCorrelatePyramid[pyr,#,NormalizedSquaredEuclideanDistance]&,
     {AlienFeature1,AlienFeature2,AlienFeature3}]];
   res=(
      MVCorrelatePyramid[AlienFeatureMaps[[1]],AlienFeature1Kernel]*
      MVCorrelatePyramid[AlienFeatureMaps[[2]],AlienFeature2Kernel]*
      MVCorrelatePyramid[AlienFeatureMaps[[3]],AlienFeature3Kernel]);
)


AlienRecognitionOutput[image_]:=( 
   AlienRecognition[image];
   Show[image//DispImage,OutlineGraphics[BoundingRectangles[res,1.0,{10,10}],Red]] );

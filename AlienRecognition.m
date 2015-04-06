(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"


(* ::Input:: *)
(*AlienModel=StandardiseImage[Import["c:/users/julian/my documents/github/machine-vision/AlienModel.jpg"]];*)


(* ::Input:: *)
(*feature1=AlienModel[[62-8;;62+8;;2,62-8;;62+8;;2]];*)
(*feature2=AlienModel[[55-8;;55+8;;2,107-8;;107+8;;2]];*)
(*Map[DispImage,{feature1,feature2}]*)


(* ::Input:: *)
(*Feature1Kernel=ConstantArray[1,{3,3}];*)
(*Feature2Kernel=Table[If[y<=-4&&x>=21,1,0],{y,-5,+5},{x,-22,+22}];Feature2Kernel//DispImage*)


(* ::Input:: *)
(*probF[z_]=FullSimplify[PDF[HalfNormalDistribution[15],z],Assumptions->z>0];*)


(* ::Input:: *)
(*AlienRecognition[image_]:=( *)
(*pyr=BuildPyramid[image];*)
(*featureMaps=probF[Map[*)
(*   MVCorrelatePyramid[pyr,#,NormalizedSquaredEuclideanDistance]&,*)
(*     {feature1,feature2}]];*)
(*res=(MVCorrelatePyramid[featureMaps[[1]],Feature1Kernel]**)
(*   MVCorrelatePyramid[featureMaps[[2]],Feature2Kernel]);*)
(*)*)


(* ::Input:: *)
(*AlienRecognitionOutput[image_]:=( *)
(*AlienRecognition[image];*)
(*Show[image//DispImage,BoundingRectangles[res,1.0,{10,10}]//OutlineGraphics] );*)

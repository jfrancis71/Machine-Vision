(* ::Package:: *)

EngSparkModel=StandardiseImage[Import["c:/users/julian/my documents/github/machine-vision/EngSparkModel.jpg"],128];


(EngSparkFeature1=EngSparkModel[[63-4;;63+4,51-8;;51+8]]);
(EngSparkFeature2=EngSparkModel[[49-4;;49+4,72-4;;72+4]]);
(EngSparkFeature3=EngSparkModel[[47-5;;47+5,31-4;;31+4]])


EngSparkFeature1Kernel=ConstantArray[1,{1,1}];
EngSparkFeature2Kernel=Table[If[y<=-13&&x>=20,1,0],{y,-15,+15},{x,-23,+23}];
EngSparkFeature3Kernel=Table[If[y<=-13&&x<=-20,1,0],{y,-15,+15},{x,-23,+23}];


EngSparkRecognition[image_]:=( 
   pyr=BuildPyramid[image];
   EngSparkFeatureMaps=probF[Map[
   MVCorrelatePyramid[pyr,#,NormalizedSquaredEuclideanDistance]&,
     {EngSparkFeature1,EngSparkFeature2,EngSparkFeature3}]];
   res=(
      MVCorrelatePyramid[EngSparkFeatureMaps[[1]],EngSparkFeature1Kernel]*
      MVCorrelatePyramid[EngSparkFeatureMaps[[2]],EngSparkFeature2Kernel]*(1/12)
      MVCorrelatePyramid[EngSparkFeatureMaps[[3]],EngSparkFeature3Kernel]*(1/12)
   )
);


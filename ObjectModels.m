(* ::Package:: *)

(* ::Text:: *)
(*Alien Model*)


AlienModel=StandardiseImage[Import["c:/users/julian/my documents/github/machine-vision/AlienModel.jpg"]];


(* Note, don't forget the step 2! *)
AlienFeature1=AlienModel[[62-8;;62+8;;2,62-8;;62+8;;2]];
AlienFeature2=AlienModel[[55-8;;55+8;;2,107-8;;107+8;;2]];
AlienFeature3=AlienModel[[34-5;;34+5,51-4;;51+4;;2]];
AlienFeatureKernels={AlienFeature1,AlienFeature2,AlienFeature3};


AlienFeature1Kernel=ConstantArray[1,{3,3}];
AlienFeature2Kernel=Table[If[y<=-4&&x>=21,1,0],{y,-5,+5},{x,-22,+22}];
AlienFeature3Kernel=Table[If[y<-13&&x<-5,1,0],{y,-18,+18},{x,-7,+7}];
AlienRelationKernels={AlienFeature1Kernel,AlienFeature2Kernel,AlienFeature3Kernel};


CokeModel=StandardiseImage[Import["c:/users/julian/my documents/github/machine-vision/CokeModel.jpg"]];


CokeFeature1=CokeModel[[14-5;;14+5,74-5;;74+5]];
CokeFeature2=CokeModel[[37-5;;37+5,74-5;;74+5]];
CokeFeatureKernels={CokeFeature1,CokeFeature2};


CokeFeature1Kernel=ConstantArray[1,{3,3}];
CokeFeature2Kernel=Table[If[y>22,1,0],{y,-25,+25},{x,-2,+2}];
CokeRelationKernels={CokeFeature1Kernel,CokeFeature2Kernel};


EngSparkModel=StandardiseImage[Import["c:/users/julian/my documents/github/machine-vision/EngSparkModel.jpg"],128];


EngSparkFeature1=EngSparkModel[[63-4;;63+4,51-8;;51+8]];
EngSparkFeature2=EngSparkModel[[49-4;;49+4,72-4;;72+4]];
EngSparkFeature3=EngSparkModel[[47-5;;47+5,31-4;;31+4]];
EngSparkFeatureKernels={EngSparkFeature1,EngSparkFeature2,EngSparkFeature3};


EngSparkFeature1Kernel=ConstantArray[1,{1,1}];
EngSparkFeature2Kernel=Table[If[y<=-13&&x>=20,1,0],{y,-15,+15},{x,-23,+23}];
EngSparkFeature3Kernel=Table[If[y<=-13&&x<=-20,1,0],{y,-15,+15},{x,-23,+23}];
EngSparkRelationKernels={EngSparkFeature1Kernel,EngSparkFeature2Kernel,EngSparkFeature3Kernel};


FaceModel=StandardiseImage[Import["c:/users/julian/secure/Shape Recognition/huttenlocher/images/faces/Training/image_0001.jpg"]
];


leftEye=ImageData[ImageTake[FaceModel//Image,{85,85}-{45,39},{35,45}]];
rightEye=ImageData[ImageTake[FaceModel//Image,{85,85}-{45,39},{53,63}]];
FaceFeatureKernels={leftEye,rightEye};


leftEyeKernel=Table[0,{1},{21}];leftEyeKernel[[1,1]]=1;leftEyeKernel[[1,2]]=1;leftEyeKernel[[1,3]]=1;
rightEyeKernel=Table[0,{1},{21}];rightEyeKernel[[1,19]]=1;rightEyeKernel[[1,20]]=1;rightEyeKernel[[1,21]]=1;
FaceRelationKernels={leftEyeKernel,rightEyeKernel};

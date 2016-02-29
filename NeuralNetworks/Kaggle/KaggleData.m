(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


data=Import["C:\\Users\\Julian\\ImageDataSetsPublic\\FacialKeypoints\\training\\training.csv"];


faceImages=Table[(Partition[ReadList[StringToStream[data[[t,31]]],Number],96]/255.),{t,2,7050}];


faceKeypoints=data[[2;;-1,1;;30]];


(*drawFace[face_,keyP_]:=
   Show[face//Image,Graphics[{Red,Point[{keyP[[1]],96-keyP[[2]]}],Point[{keyP[[3]],96-keyP[[4]]}]}]]*)


kaggleImages=Map[Reverse[ImageData[ImageResize[Image[#],{32,32}]]]&,faceImages];


RawTrainingLabels=faceKeypoints[[All,1;;4]]*32/96;


dirty=Position[Map[VectorQ[#,MachineNumberQ]&,RawTrainingLabels],False];


procImages=Delete[kaggleImages,dirty];
procLabels=Delete[RawTrainingLabels,dirty];


SeedRandom[1234];
samp=RandomSample[Transpose[{procImages,procLabels}]];


drawFace[face_,keyP_]:=
Show[face//DispImage,Graphics[{Red,Point[{keyP[[1]],32-keyP[[2]]}],Point[{keyP[[3]],32-keyP[[4]]}]}]]


FaceImages=samp[[All,1]];
FaceLabels=Map[{#[[1]],#[[2]],#[[3]],#[[4]]}&,samp[[All,2]]];

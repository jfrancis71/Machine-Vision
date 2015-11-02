(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


data=Import["C:\\Users\\Julian\\ImageDataSetsPublic\\FacialKeypoints\\training\\training.csv"];


faceImages=Table[(Partition[ReadList[StringToStream[data[[t,31]]],Number],96]/255.),{t,2,7049}];


drawFace[face_,keyP_]:=
   Show[face//Image,Graphics[{Red,Point[{keyP[[1]],96-keyP[[2]]}],Point[{keyP[[3]],96-keyP[[4]]}]}]]


images=ReadImagesFromDirectory["C:\\Users\\julian\\Google Drive\\Personal\\Pictures\\Coolpix\\*\\"];


patches=Flatten[Map[Partition[#,{32,32}]&,images],2];


kaggleImages=Map[Reverse[ImageData[ImageResize[Image[#],{32,32}]]]&,faceImages];


RawTrainingImages=Join[kaggleImages,patches];


RawTrainingLabels=Join[ConstantArray[1,Length[kaggleImages]],ConstantArray[0,Length[patches]]];


SeedRandom[1234];
samp=RandomSample[Transpose[{RawTrainingImages,RawTrainingLabels}]];


FaceImages=samp[[All,1]];
FaceLabels=Map[{#}&,samp[[All,2]]];

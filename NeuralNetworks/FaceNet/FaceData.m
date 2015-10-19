(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


LFWDir="C:\\Users\\julian\\ImageDataSetsPublic\\LFW\\lfw\\";


LFW=Join[
   ReadImagesFromDirectory[LFWDir<>"\\A*\\",32],
   ReadImagesFromDirectory[LFWDir<>"\\B*\\",32],
   ReadImagesFromDirectory[LFWDir<>"\\C*\\",32],
   ReadImagesFromDirectory[LFWDir<>"\\D*\\",32],
   ReadImagesFromDirectory[LFWDir<>"\\E*\\",32],
   ReadImagesFromDirectory[LFWDir<>"\\F*\\",32],
   ReadImagesFromDirectory[LFWDir<>"\\G*\\",32],
   ReadImagesFromDirectory[LFWDir<>"\\H*\\",32],
   ReadImagesFromDirectory[LFWDir<>"\\I*\\",32]
];


images=ReadImagesFromDirectory["C:\\Users\\Julian\\Google Drive\\Personal\\Pictures\\Iphone Pictures\\**\\"];


patches=Flatten[Map[Partition[#,{32,32}]&,images],2];


RawTrainingImages=Join[LFW[[1;;5000]],patches[[1;;5000]]];


RawTrainingLabels=Join[ConstantArray[1,5000],ConstantArray[0,5000]];


SeedRandom[1234];
samp=RandomSample[Transpose[{RawTrainingImages,RawTrainingLabels}]];


FaceImages=samp[[All,1]];
FaceLabels=Map[{#}&,samp[[All,2]]];

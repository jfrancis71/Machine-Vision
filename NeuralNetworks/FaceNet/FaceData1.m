(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


LFWDir="C:\\Users\\julian\\ImageDataSetsPublic\\LFW\\lfw\\";


LFW=Join[
   ReadImagesFromDirectory[LFWDir<>"\\A*\\",32],
   ReadImagesFromDirectory[LFWDir<>"\\B*\\",32],
   ReadImagesFromDirectory[LFWDir<>"\\C*\\",32]
];


(* corresponds to 1054 total faces, now 2515 *)


images=ReadImagesFromDirectory["C:\\Users\\Julian\\Google Drive\\Personal\\Pictures\\Iphone Pictures\\**\\"];


patches=Flatten[Map[Partition[#,{32,32}]&,images],2];


(* There are 2512 patches, now 5688 *)


RawTrainingImages=Join[LFW,patches[[1;;2500]]];


RawTrainingLabels=Join[ConstantArray[1,LFW//Length],ConstantArray[0,patches[[1;;2500]]//Length]];


SeedRandom[1234];
samp=RandomSample[Transpose[{RawTrainingImages,RawTrainingLabels}]];


FaceImages=samp[[All,1]];
FaceLabels=Map[{#}&,samp[[All,2]]];

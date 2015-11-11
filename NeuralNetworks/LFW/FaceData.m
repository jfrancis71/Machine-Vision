(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


LFWDir="C:\\Users\\julian\\ImageDataSetsPublic\\LFW\\lfw\\";


LFW=ReadImagesFromDirectory[LFWDir<>"\\*\\",32];


images=ReadImagesFromDirectory["C:\\Users\\julian\\Google Drive\\Personal\\Pictures\\Coolpix\\*\\"];


patches=Flatten[Map[Partition[#,{32,32}]&,images],2];


(* Very small imbalance in favour of LWF. Approx 13,000 positives. Training around 26,000 *)


RawTrainingImages=Join[LFW,patches];


RawTrainingLabels=Join[ConstantArray[1,Length[LFW]],ConstantArray[0,Length[patches]]];


SeedRandom[1234];
samp=RandomSample[Transpose[{RawTrainingImages,RawTrainingLabels}]];


FaceImages=samp[[All,1]];
FaceLabels=Map[{#}&,samp[[All,2]]];

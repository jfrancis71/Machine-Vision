(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


FaceScrubDir="C:\\Users\\julian\\ImageDataSetsPublic\\NFaceScrub\\";


males=ReadImagesFromDirectory[FaceScrubDir<>"ActorFaces\\",32];


females=ReadImagesFromDirectory[FaceScrubDir<>"ActressFaces\\",32];


images=ReadImagesFromDirectory["C:\\Users\\julian\\ImageDataSetsPublic\\Distractors\\*"];


patches=Flatten[Map[Partition[#,{32,32}]&,images],2];


(* Very small imbalance in favour of LWF. Approx 13,000 positives. Training around 26,000 *)


RawTrainingImages=Join[males,females,patches];


RawTrainingLabels=Join[ConstantArray[1,Length[males]+Length[females]],ConstantArray[0,Length[patches]]];


SeedRandom[1234];
samp=RandomSample[Transpose[{RawTrainingImages,RawTrainingLabels}]];


FaceImages=samp[[All,1]];
FaceLabels=Map[{#}&,samp[[All,2]]];

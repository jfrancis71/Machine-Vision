(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


images=ReadImagesFromDirectory["C:\\Users\\julian\\ImageDataSetsPublic\\Distractors\\*"];


patches=Flatten[Map[Partition[#,{32,32}]&,images],2];


SeedRandom[1234];
randomPatches=Table[Random[],{120000},{32},{32}];


RawTrainingImages=Join[patches,randomPatches];


RawTrainingLabels=Join[ConstantArray[1,Length[patches]],ConstantArray[0,Length[randomPatches]]];


SeedRandom[1234];
samp=RandomSample[Transpose[{RawTrainingImages,RawTrainingLabels}]];


discrimImages=samp[[All,1]];
discrimLabels=Map[{#}&,samp[[All,2]]];

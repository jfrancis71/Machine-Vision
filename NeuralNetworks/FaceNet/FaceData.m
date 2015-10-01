(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


LFWADir="C:\\Users\\julian\\ImageDataSetsPublic\\LFW-A\\lfw2\\A*";


files=FileNames[LFWADir<>"\\*.jpg"];


(* corresponds to 1054 total faces *)


images=ReadImagesFromDirectory["C:\\Users\\Julian\\Google Drive\\Personal\\Pictures\\Iphone Pictures\\30062014"];


patches=Flatten[Map[Partition[#,{32,32}]&,images],2];


(* There are 2512 patches *)


RawTrainingImages=Join[LFW,patches[[1;;1000]]];


RawTrainingLabels=Join[ConstantArray[1,LFW//Length],ConstantArray[0,patches[[1;;1000]]//Length]];


SeedRandom[1234];
samp=RandomSample[Transpose[{RawTrainingImages,RawTrainingLabels}]];


FaceImages=samp[[All,1]];
FaceLabels=Map[{#}&,samp[[All,2]]];

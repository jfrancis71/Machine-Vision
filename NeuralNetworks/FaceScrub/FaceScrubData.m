(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


FaceScrubDir="C:\\Users\\julian\\ImageDataSetsPublic\\NFaceScrub\\";


males=ReadImagesFromDirectory[FaceScrubDir<>"ActorFaces\\*",32];


females=ReadImagesFromDirectory[FaceScrubDir<>"ActressFaces\\*",32];


images=ReadImagesFromDirectory["C:\\Users\\julian\\ImageDataSetsPublic\\Distractors\\*"];


patches=Flatten[Map[Partition[#,{32,32}]&,images],2];


(* Very small imbalance in favour of LWF. Approx 13,000 positives. Training around 26,000 *)


patchesWithoutFaces=Delete[patches,Map[{#}&,{49634,48354,49762,7977,49179,64403,49115,49170,49154,49107,50179,49603,48395,65434,48587,48067,49355,62299,48315,51275,50611,64572,49651,50451,62100,78410,49803,68203,73265,50818,48226,50583,49507,49799,50291,49290,64259,50331,50770,63439,51170,8519,21211,48722,50162,48763,51307,49546,49719,51107,70726,49067,49099,50411,49562,9985,51283,48339,63271,20983,48331,66946,48555}]];


RawTrainingImages=Join[males,females,patchesWithoutFaces];


RawTrainingLabels=Join[ConstantArray[1,Length[males]+Length[females]],ConstantArray[0,Length[patchesWithoutFaces]]];


SeedRandom[1234];
samp=RandomSample[Transpose[{RawTrainingImages,RawTrainingLabels}]];


FaceImages=samp[[All,1]];
FaceLabels=Map[{#}&,samp[[All,2]]];

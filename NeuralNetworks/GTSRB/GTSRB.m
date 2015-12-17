(* ::Package:: *)

files=FileNames["C:\\Users\\julian\\ImageDataSetsPublic\\GTSRB\\GTSRB\\Final_Training\\Images\\00017\\*.ppm"];Length[files]


res=Map[Import,files];


trafficSigns=Map[Reverse[ImageData[ImageResize[ColorConvert[#,"GrayScale"],{32,32}]]]&,res];


Length[trafficSigns]


str=Import["C:\\Users\\julian\\ImageDataSetsPublic\\GTSRB\\GTSRB\\Final_Training\\Images\\00017\\"<>"GT-00017.csv"][[2;;-1,1]];Length[str]


(*Need NNImageTrim function from FaceScrub Generate*)


SeedRandom[1234];


Dynamic[{e,s}]


rdDat=Table[ReadList[StringToStream[str[[s]]],Word,WordSeparators->{";"}],{s,1,Length[str]}];
ptDat=Table[Import["C:\\Users\\julian\\ImageDataSetsPublic\\GTSRB\\GTSRB\\Final_Training\\Images\\00017\\"<>rdDat[[s,1]]],{s,1,Length[str]}];


trafficSignsRep=Table[
rd=rdDat[[s]];
pt=ptDat[[s]];
cols=Map[FromDigits,rd[[4;;7]]];
bb=cols;
size=bb[[3]]-bb[[1]];
   centx=(bb[[1]]+bb[[3]])/2;
   centy=(bb[[2]]+bb[[4]])/2;
   {centx,centy}={centx,centy}+({Random[],Random[]}-{.5,.5})*size*.25;
   size=size*(1+(Random[]-.5)*.5);
   img=pt;
   ImageData[ColorConvert[ImageResize[NNImageTrim[img,
         {{centx-(size/2),centy-(size/2)},{centx+(size/2),centy+(size/2)}}
         ],{32,32}],"GrayScale"],DataReversed->True],
{e,1,100},
{s,1,Length[str]}];


trafficSigns=Flatten[trafficSignsRep,1];


images=ReadImagesFromDirectory["C:\\Users\\julian\\ImageDataSetsPublic\\Distractors\\*"];


patches=Flatten[Map[Partition[#,{32,32}]&,images],2];


Length[patches]


RawTrainingImages=Join[trafficSigns,patches];


RawTrainingLabels=Join[ConstantArray[1,Length[trafficSigns]],ConstantArray[0,Length[patches]]];


SeedRandom[1234];
samp=RandomSample[Transpose[{RawTrainingImages,RawTrainingLabels}]];


GTSRBImages=samp[[All,1]];
GTSRBLabels=Map[{#}&,samp[[All,2]]];


SeedRandom[1234];
GTSRBNet={
   PadFilter[2],Convolve2DToFilterBankInit[32,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[32,32,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[32,64,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   Adaptor3DTo1D[64,4,4],
   FullyConnected1DTo1DInit[64*4*4,1],
   Logistic
};


wl=GTSRBNet;
TrainingHistory={};
ValidationHistory={};
\[Lambda]=.01;


Length[GTSRBImages]


TrainGTSRBNet:=MiniBatchGradientDescent[
      wl,GTSRBImages[[1;;-1000]],GTSRBLabels[[1;;-1000]],
      NNGrad,CrossEntropyLoss,
        {MaxEpoch->500000,
         ValidationInputs->GTSRBImages[[-1000;;-1]],
         ValidationTargets->GTSRBLabels[[-1000;;-1]],
         StepMonitor->NNCheckpoint["GTSRB\\GTSRB"],
         Momentum->.9,MomentumType->"Nesterov",
         InitialLearningRate->\[Lambda]}];

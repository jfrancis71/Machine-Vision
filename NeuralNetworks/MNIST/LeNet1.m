(* ::Package:: *)

Needs["Developer`"]


(*
a is a vector
*)
SoftMax[a_]:=E^a/Total[E^a]


(*
D2ConvolveA calculates the A output of the next layer of a neural network assuming
  inputs: kxk matrix
weights : 26 convolution kernel, m is the mxm dimensions of output layer
*)
D2ConvolveA[inputs_,weights_]:=ListCorrelate[Partition[Rest[weights],5],inputs,{3,3},0]+First[weights]


Subsample[inputs_]:=ToPackedArray[
Table[inputs[[j*2,i*2]],{j,1,Length[inputs]/2},{i,1,Length[inputs]/2}]]


(*
BlockD2ConvolveA calculates the A output of the next layer of a neural network assuming
  inputs: nxkxk matrix
weights : 26 convolution kernel, m is the mxm dimensions of output layer
*)
BlockD2ConvolveA[inputs_,weights_]:=ListCorrelate[{Partition[Rest[weights],5]},inputs,{1,3,3},0]+First[weights]


(*
   BlockFeatureMapD2ConvolveA
  featureMaps: n*12*m*m where m might be 16
weightMaps: 12*26
*)
BlockFeatureMapD2ConvolveA[featureMaps_,weightMaps_]:=
Total[MapThread[BlockD2ConvolveA,{Transpose[featureMaps,{2,1,3,4}],weightMaps}]]


A[inputs_?VectorQ,weights_?MatrixQ]:= 
weights.Prepend[inputs,1]


ForwardPropogation[inputs_,w_]:=(
H1A=Table[Subsample[D2ConvolveA[input,w[[1,f]]]],{input,inputs},{f,1,12}];

H1=ToPackedArray[ArcTan[H1A]];

Timer["H2A=",H2A=Map[Subsample,Transpose[Map[BlockFeatureMapD2ConvolveA[H1,#]&,w[[2]]],{2,1,3,4}],{2}];];
H2 = ArcTan[H2A];

H3A=Table[A[Flatten[H2[[n]]],w[[3]]],{n,1,Length[inputs]}];
H3=ArcTan[H3A];

OutputLayer=Table[A[H3[[n]],w[[4]]],{n,1,Length[inputs]}]
);


w=Import["C:\\Users\\Julian\\secure\\Shape Recognition\\MNIST\\Categorical\\LetNet1Weights1.wdx"];


ReadMINSTImageFile[file_] := (
   RawImageFileData=BinaryReadList[file,"Byte",200000000];
   RawImageStream=RawImageFileData[[17;;-1]];
   RawImages=Partition[RawImageStream,28*28];
   Map[If[#1 > 128, 1, 0] &,Map[Partition[#,28]&,RawImages],{3}]
)

ReadMINSTLabelFile[file_] := (
   RawClassificationFileData=BinaryReadList[file,"Byte",200000000];
   RawLabelStream=RawClassificationFileData[[9;;-1]]
)

TrainingImages = ReadMINSTImageFile["c:/users/julian/downloads/train-images-idx3-ubyte"];
TestImages = ReadMINSTImageFile["c:/users/julian/downloads/t10k-images-idx3-ubyte"];

TrainingLabels = ReadMINSTLabelFile["c:/users/julian/downloads/train-labels-idx1-ubyte"];
TestLabels = ReadMINSTLabelFile["c:/users/julian/downloads/t10k-labels-idx1-ubyte"];


RTrainingImages=Map[ImageData[Binarize[ImageResize[Image[#],{16,16}]]]&,TrainingImages[[1;;100]]];


RTestImages=Map[ImageData[Binarize[ImageResize[Image[#],{16,16}]]]&,TestImages];

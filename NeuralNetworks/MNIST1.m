(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


ReadMINSTImageFile[file_] := (
   RawImageFileData=BinaryReadList[file,"Byte",200000000];
   RawImageStream=RawImageFileData[[17;;-1]];
   RawImages=Partition[RawImageStream,28*28];
   Map[Reverse,Map[If[#1 > 128, 1, 0] &,Map[Partition[#,28]&,RawImages],{3}]]
)

ReadMINSTLabelFile[file_] := (
   RawClassificationFileData=BinaryReadList[file,"Byte",200000000];
   RawLabelStream=RawClassificationFileData[[9;;-1]]
)

TrainingImages = ReadMINSTImageFile["C:\\Users\\Julian\\ImageDataSetsPublic\\MNIST\\train-images-idx3-ubyte"];
TestImages = ReadMINSTImageFile["C:\\Users\\Julian\\ImageDataSetsPublic\\MNIST\\t10k-images-idx3-ubyte"];

TrainingLabels = ReadMINSTLabelFile["C:\\Users\\Julian\\ImageDataSetsPublic\\MNIST\\train-labels-idx1-ubyte"];
TestLabels = ReadMINSTLabelFile["C:\\Users\\Julian\\ImageDataSetsPublic\\MNIST\\t10k-labels-idx1-ubyte"];


pos=Position[TrainingLabels,2|3];
r1=RandomList[[1;;28*28]]-.5;
MN1Network={Adaptor2DTo1D[28],FullyConnected1DTo1D[{0},{r1}/7840.]};
MN1Inputs=Extract[TrainingImages,pos]-0.0;
MN1Outputs=Extract[TrainingLabels,pos]-2.0;
MN1Trained:=AdaptiveGradientDescent[MN1Network,MN1Inputs,MN1Outputs,Grad,Loss1D,500000];

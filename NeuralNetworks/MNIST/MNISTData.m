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


(*func is a function which takes an array of size*size and
returns an output to be displayed *)

DrawingPad4[func_,size_Integer] := (
  m = Table[0, {i, size}, {j, size}];
  DynamicModule[{},
    Row[{Button["Reset", m = Table[0, {i, size}, {j, size}]],
      EventHandler[
       Dynamic@ArrayPlot[
         m], 
       "MouseDragged" :> (pt = Floor[MousePosition["Graphics"]] + {1, 1};
         (*pt=Map[Min[size,#]&,pt];*)
         pt = {Min[size, pt[[1]]], Min[size, pt[[2]]]};
         (ci = {size - pt[[2]], pt[[1]]});
         m[[ci[[1]], ci[[2]]]] = 1)
       ], Dynamic@func[m]}]
  ]
)


GUI:=DrawingPad4[ForwardPropogation[{#//Reverse},wl]&,28]

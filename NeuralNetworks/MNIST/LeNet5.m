(* ::Package:: *)

(*
   See Yann Le Cunn et al 1998:

http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

Network Architecture: Convolutional Network, see Figure 2 in above paper

They claim .9% test result

Our Results:
Iteration: 
Training Loss: .
Training Classification: .%

Validation Loss: .
Validation Classification: .0%
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/MNIST/MNISTData.m"


MNISTLeNet5={
   Convolve2DToFilterBank[{
      Convolve2D[0,Partition[Table[Random[],{25}]-.5,5]/25],
      Convolve2D[0,Partition[Table[Random[],{25}]-.5,5]/25],
      Convolve2D[0,Partition[Table[Random[],{25}]-.5,5]/25],
      Convolve2D[0,Partition[Table[Random[],{25}]-.5,5]/25],
      Convolve2D[0,Partition[Table[Random[],{25}]-.5,5]/25],
      Convolve2D[0,Partition[Table[Random[],{25}]-.5,5]/25]
}],Tanh,
   MaxPoolingFilterBankToFilterBank,
   ConvolveFilterBankToFilterBank[Table[
      ConvolveFilterBankTo2D[0,{
         Partition[Table[Random[],{25}]-.5,5]/25,
         Partition[Table[Random[],{25}]-.5,5]/25,
         Partition[Table[Random[],{25}]-.5,5]/25,
         Partition[Table[Random[],{25}]-.5,5]/25,
         Partition[Table[Random[],{25}]-.5,5]/25,
         Partition[Table[Random[],{25}]-.5,5]}/25],
      {f,0,15}]
   ],Tanh,
   MaxPoolingFilterBankToFilterBank,
   Adaptor3DTo1D[16,5,5],
   FullyConnected1DTo1D[Table[Random[],{10}],(Partition[Table[Random[],{10*400}],400]-.5)/4000],
   Softmax
};


MNISTLeNet5TrainingInputs=Map[ArrayPad[#,{2,2}]&,TrainingImages[[1;;500]]*1.];
MNISTLeNet5TrainingOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels[[1;;500]]];

MNISTLeNet5ValidationInputs=Map[ArrayPad[#,{2,2}]&,TrainingImages[[501;;600]]*1.];
MNISTLeNet5ValidationOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels[[501;;600]]];


wl=MNISTLeNet5;


 MNISTLeNet5Train:=AdaptiveGradientDescent[
   wl,MNISTLeNet5TrainingInputs,MNISTLeNet5TrainingOutputs,
   Grad,ClassificationLoss,
     {MaxLoop->500000,
      ValidationInputs->MNISTLeNet5ValidationInputs,
      ValidationTargets->MNISTLeNet5ValidationOutputs}];

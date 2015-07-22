(* ::Package:: *)

(*
   See Yann Le Cunn et al 1995:

http://yann.lecun.com/exdb/publis/pdf/lecun-95b.pdf

Network Architecture: Large Fully Connected Multi-Layer Neural Network
                      400-300-10

They claim 1.6% test result

Our Results:
Iteration: 
Training Loss: .
Training Classification: .%

Validation Loss: .
Validation Classification: .0%
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/MNIST/MNISTData.m"


MNISTLeNet1={
   Convolve2DToFilterBank[{
      Convolve2D[0,Partition[Table[Random[],{25}]-.5,5]/25],
      Convolve2D[0,Partition[Table[Random[],{25}]-.5,5]/25],
      Convolve2D[0,Partition[Table[Random[],{25}]-.5,5]/25],
      Convolve2D[0,Partition[Table[Random[],{25}]-.5,5]/25]
}],
   MaxPoolingFilterBankToFilterBank,
   ConvolveFilterBankToFilterBank[Table[
      ConvolveFilterBankTo2D[0,{
         Partition[Table[Random[],{25}]-.5,5]/25,
         Partition[Table[Random[],{25}]-.5,5]/25,
         Partition[Table[Random[],{25}]-.5,5]/25,
         Partition[Table[Random[],{25}]-.5,5]}/25],
      {f,0,11}]
   ],
   MaxPoolingFilterBankToFilterBank,
   Adaptor3DTo1D[12,4,4],
   FullyConnected1DTo1D[Table[Random[],{10}],(Partition[Table[Random[],{10*192}],192]-.5)/1920],
   Softmax
};


MNISTLeNet1TrainingInputs=TrainingImages[[1;;500,1;;28,1;;28]]*1.;
MNISTLeNet1TrainingOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels[[1;;500]]];

MNISTLeNet1ValidationInputs=TrainingImages[[501;;600,1;;28,1;;28]]*1.;
MNISTLeNet1ValidationOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels[[501;;600]]];


 MNISTLeNet1Train:=AdaptiveGradientDescent[
   MNISTLeNet1,MNISTLeNet1TrainingInputs,MNISTLeNet1TrainingOutputs,
   Grad,ClassificationLoss,
     {MaxLoop->500000,
      ValidationInputs->MNISTLeNet1ValidationInputs,
      ValidationTargets->MNISTLeNet1ValidationOutputs}];

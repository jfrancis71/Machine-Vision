(* ::Package:: *)

(*
   See Yann Le Cunn et al 1995:

http://yann.lecun.com/exdb/publis/pdf/lecun-95b.pdf

Network Architecture: Convolutional Network, see Figure 1 in above paper

They claim 1.7% test result

Our Results: Needs more training (note the training set)
Iteration: Around 2,000 four days
Training Loss: .251
Training Classification: 92.3%

Validation Loss: .295
Validation Classification: 90.8%
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/MNIST/MNISTData.m"


SeedRandom[1234];
MNISTLeNet1={
   Convolve2DToFilterBank[{
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
         Partition[Table[Random[],{25}]-.5,5]/25}/4],
      {f,0,11}]
   ],Tanh,
   MaxPoolingFilterBankToFilterBank,
   Adaptor3DTo1D[12,4,4],
   FullyConnected1DTo1D[Table[Random[],{10}],(Partition[Table[Random[],{10*192}],192]-.5)/1920],
   Softmax
};


MNISTLeNet1TrainingInputs=TrainingImages[[1;;5000,1;;28,1;;28]]*1.;
MNISTLeNet1TrainingOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels[[1;;5000]]];

MNISTLeNet1ValidationInputs=TrainingImages[[50001;;55000,1;;28,1;;28]]*1.;
MNISTLeNet1ValidationOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels[[50001;;55000]]];


wl=MNISTLeNet1;
TrainingHistory={};
ValidationHistory={};


 MNISTLeNet1Train:=AdaptiveGradientDescent[
   wl,MNISTLeNet1TrainingInputs,MNISTLeNet1TrainingOutputs,
   Grad,ClassificationLoss,
     {MaxLoop->500000,
      ValidationInputs->MNISTLeNet1ValidationInputs,
      ValidationTargets->MNISTLeNet1ValidationOutputs}];

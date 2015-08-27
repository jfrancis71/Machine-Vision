(* ::Package:: *)

(*
   See Yann Le Cunn et al 1995:

http://yann.lecun.com/exdb/publis/pdf/lecun-95b.pdf

Network Architecture: Convolutional Network, see Figure 1 in above paper

They claim 1.7% test result

Our Results: (Training set size, 10,000)
Iteration: 991, approx 12 hours
Training Loss: .025
Training Classification: 99.6%

Validation Loss: .096
Validation Classification: 97.2%
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


MNISTLeNet1TrainingInputs=TrainingImages[[1;;10000,1;;28,1;;28]]*1.;
MNISTLeNet1TrainingOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels[[1;;10000]]];

MNISTLeNet1ValidationInputs=TrainingImages[[50001;;55000,1;;28,1;;28]]*1.;
MNISTLeNet1ValidationOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels[[50001;;55000]]];


LeNet1Train:=(
   name="MNIST\\LeNet1";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   AdaptiveGradientDescent[
      wl,MNISTLeNet1TrainingInputs,MNISTLeNet1TrainingInputs,
      NNGrad,ClassificationLoss,
        {MaxLoop->500000,
         UpdateFunction->CheckpointWebMonitor[name,100],
         ValidationInputs->MNISTLeNet1ValidationInputs,
         ValidationTargets->MNISTLeNet1ValidationOutputs,
         InitialLearningRate->\[Lambda]}];
)

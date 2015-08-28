(* ::Package:: *)

(*
   See Yann Le Cunn et al 1998:

http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

Network Architecture: Convolutional Network, see Figure 2 in above paper
I think we're missing a final convolutional layer, not sure what it did in original paper
as layer too small to fit filter.

They claim .9% test result

Our Results:
Iteration: 421
Training Loss: .084
Training Classification: 97.7%

Validation Loss: .118
Validation Classification: 96.7%
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
      {f,1,16}]
   ],Tanh,
   MaxPoolingFilterBankToFilterBank,
   Adaptor3DTo1D[16,5,5],
   FullyConnected1DTo1D[Table[Random[],{10}],(Partition[Table[Random[],{10*400}],400]-.5)/4000],
   Softmax
};


MNISTLeNet5TrainingInputs=Map[ArrayPad[#,{2,2}]&,TrainingImages[[1;;10000]]*1.];
MNISTLeNet5TrainingOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels[[1;;10000]]];

MNISTLeNet5ValidationInputs=Map[ArrayPad[#,{2,2}]&,TrainingImages[[50001;;55000]]*1.];
MNISTLeNet5ValidationOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels[[50001;;55000]]];


LeNet5Train:=(
   name="MNIST\\LeNet5";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin[GITBaseDir,name,".wdx"]];
   AdaptiveGradientDescent[
      wl,MNISTLeNet5TrainingInputs,MNISTLeNet5TrainingOutputs,
      NNGrad,ClassificationLoss,
        {MaxLoop->500000,
         UpdateFunction->CheckpointWebMonitor[name,5],
         ValidationInputs->MNISTLeNet5ValidationInputs,
         ValidationTargets->MNISTLeNet5ValidationOutputs,
         InitialLearningRate->\[Lambda]}];
)

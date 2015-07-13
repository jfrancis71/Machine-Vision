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
      Convolve2D[0,Partition[RandomList[[1;;25]]-.5,5]],
      Convolve2D[0,Partition[RandomList[[101;;125]]-.5,5]],
      Convolve2D[0,Partition[RandomList[[201;;225]]-.5,5]],
      Convolve2D[0,Partition[RandomList[[301;;325]]-.5,5]]
}],
   MaxPoolingFilterBankToFilterBank,
   ConvolveFilterBankToFilterBank[Table[
      ConvolveFilterBankTo2D[0,{
         Partition[RandomList[[f+1;;f+25]]-.5,5],
         Partition[RandomList[[1+f+1;;1+f+25]]-.5,5],
         Partition[RandomList[[2+f+1;;2+f+25]]-.5,5],
         Partition[RandomList[[3+f+1;;3+f+25]]-.5,5]}],
      {f,0,11}]
   ],
   MaxPoolingFilterBankToFilterBank,
   Adaptor3DTo1D[12,4,4],
   FullyConnected1DTo1D[RandomList[[1;;10]],Partition[RandomList[[1;;10*192]],192]]
};


MNISTLeNet1TrainingInputs=TrainingImages[[1;;5,1;;28,1;;28]]*1.;
MNISTLeNet1TrainingOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels[[1;;5]]];

MNISTLeNet1ValidationInputs=TrainingImages[[5001;;6000,1;;28,1;;28]]*1.;
MNISTLeNet1ValidationOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels[[5001;;6000]]];


 MNISTLeNet1Trained:=AdaptiveGradientDescent[
   MNISTLeNet1,MNISTLeNet1TrainingInputs,MNISTLeNet1TrainingOutputs,
   Grad,Loss1D,
     {MaxLoop->500000,
      ValidationInputs->MNISTLeNet1ValidationInputs,
      ValidationTargets->MNISTLeNet1ValidationOutputs}];

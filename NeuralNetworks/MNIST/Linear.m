(* ::Package:: *)

(*
   Technique reviewed by Yann Le Cunn et al 1995:

http://yann.lecun.com/exdb/publis/pdf/lecun-95b.pdf

Network Architecture: Linear Network
                      400-10

They claim 8.4% test result

Our Results:
Iteration: 5178
Training Loss: .462
Training Classification: 84.1%

Validation Loss: .446
Validation Classification: 85.7%
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/MNIST/MNISTData.m"


MNISTLinearNetwork={Adaptor2DTo1D[20],FullyConnected1DTo1D[ConstantArray[0.,10],Partition[RandomList[[1;;4000]]/4000.,400]]};


MNISTLinearTrainingInputs=TrainingImages[[1;;50000,5;;24,5;;24]]*1.;
MNISTLinearTrainingOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels[[1;;50000]]];

MNISTLinearValidationInputs=TrainingImages[[50001;;60000,5;;24,5;;24]]*1.;
MNISTLinearValidationOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels[[50001;;60000]]];


MNISTLinearTrain:=AdaptiveGradientDescent[MNISTLinearNetwork,MNISTLinearTrainingInputs,MNISTLinearTrainingOutputs,Grad,Loss1D,
   {MaxLoop->500000,ValidationInputs->MNISTLinearValidationInputs,ValidationTargets->MNISTLinearValidationOutputs}];

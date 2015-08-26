(* ::Package:: *)

(*
   Technique reviewed by Yann Le Cunn et al 1995:

http://yann.lecun.com/exdb/publis/pdf/lecun-95b.pdf

Network Architecture: Linear Network
                      400-10

They claim 8.4% test result (i.e. 91.6% correct)

Our Results:
Iteration: 332
Training Loss: .339
Training Classification: 90.5%

Validation Loss: .320
Validation Classification: 91.1%
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/MNIST/MNISTData.m"


SeedRandom[1234];
MNISTLinearNetwork={
   Adaptor2DTo1D[20],
   FullyConnected1DTo1D[ConstantArray[0.,10],Partition[Table[Random[],{4000}]/4000.,400]],
   Softmax};


MNISTLinearTrainingInputs=TrainingImages[[1;;50000,5;;24,5;;24]]*1.;
MNISTLinearTrainingOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels[[1;;50000]]];

MNISTLinearValidationInputs=TrainingImages[[50001;;60000,5;;24,5;;24]]*1.;
MNISTLinearValidationOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels[[50001;;60000]]];


MNISTLinearTrain:=AdaptiveGradientDescent[MNISTLinearNetwork,MNISTLinearTrainingInputs,MNISTLinearTrainingOutputs,Grad,ClassificationLoss,
   {MaxLoop->500000,ValidationInputs->MNISTLinearValidationInputs,ValidationTargets->MNISTLinearValidationOutputs}];

(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/MNIST/MNISTData.m"


MNISTLinearNetwork={Adaptor2DTo1D[20],FullyConnected1DTo1D[ConstantArray[0.,10],Partition[RandomList[[1;;4000]]/4000.,400]]};


MNISTLinearTrainingInputs=TrainingImages[[1;;50000,5;;24,5;;24]]*1.;
MNISTLinearTrainingOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels[[1;;50000]]];


MNISTLinearTrained:=AdaptiveGradientDescent[MNISTLinearNetwork,MNISTLinearTrainingInputs,MNISTLinearTrainingOutputs,Grad,Loss1D,500000];

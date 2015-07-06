(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/MNIST/MNISTData.m"


MNISTAutoencoderTrainingInputs=Map[Flatten,TrainingImages[[1;;500,5;;24,5;;24]]*1.];


H1=50;
B=10;
AutoencoderNetwork={
   FullyConnected1DTo1D[
      ConstantArray[0,H1],Table[Random[],{H1},{400}]-.5],
   FullyConnected1DTo1D[
      ConstantArray[0,B],Table[Random[],{B},{H1}]-.5],
   FullyConnected1DTo1D[
      ConstantArray[0,H1],Table[Random[],{H1},{B}]-.5],
   FullyConnected1DTo1D[
      ConstantArray[0,400],Table[Random[],{400},{H1}]-.5]
};


wl=AutoencoderNetwork;


MNISTAutoencoderTrained:=AdaptiveGradientDescent[wl,MNISTAutoencoderTrainingInputs,MNISTAutoencoderTrainingInputs,Grad,Loss2D,{MaxLoop->500000}];

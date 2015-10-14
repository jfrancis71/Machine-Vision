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
   Convolve2DToFilterBankInit[4,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   ConvolveFilterBankToFilterBankInit[4,12,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   Adaptor3DTo1D[12,4,4],
   FullyConnected1DTo1DInit[12*4*4,10],
   Softmax
};


wl=MNISTLeNet1;
TrainingHistory={};
ValidationHistory={};
\[Lambda]=.01;


TrainLeNet1:=MiniBatchGradientDescent[
      wl,TrainingImages,TrainingTargets,
      NNGrad,ClassificationLoss,
        {MaxEpoch->500000,
         ValidationInputs->ValidationImages,
         ValidationTargets->ValidationTargets,
         StepMonitor->NNCheckpoint["MNIST\\LeNet1"],
         InitialLearningRate->\[Lambda]}];

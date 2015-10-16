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
   PadFilter[2],
   Convolve2DToFilterBankInit[6,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   ConvolveFilterBankToFilterBankInit[6,16,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   Adaptor3DTo1D[16,5,5],
   FullyConnected1DTo1DInit[16*5*5,120],Tanh,
   FullyConnected1DTo1DInit[120,84],Tanh,
   FullyConnected1DTo1DInit[84,10],
   Softmax
};


wl=MNISTLeNet5;
TrainingHistory={};
ValidationHistory={};
\[Lambda]=.01;


TrainLeNet5:=MiniBatchGradientDescent[
      wl,TrainingImages,MNISTTrainingTargets,
      NNGrad,ClassificationLoss,
        {MaxEpoch->500000,
         ValidationInputs->ValidationImages,
         ValidationTargets->MNISTValidationTargets,
         StepMonitor->NNCheckpoint["MNIST\\LeNet5"],
         InitialLearningRate->\[Lambda]}];

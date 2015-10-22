(* ::Package:: *)

(*
   See Yann Le Cunn et al 1995:

http://yann.lecun.com/exdb/publis/pdf/lecun-95b.pdf

Network Architecture: Convolutional Network, see Figure 1 in above paper

They claim 1.7% test result

Our Results:
Iteration: 200
Training Loss: .044
Training Classification: 1.4%

Validation Loss: .139
Validation Classification: 1.8%
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
      wl,TrainingImages,MNISTTrainingTargets,
      NNGrad,ClassificationLoss,
        {MaxEpoch->200,
         ValidationInputs->ValidationImages,
         ValidationTargets->MNISTValidationTargets,
         StepMonitor->NNCheckpoint["MNIST\\LeNet1"],
         InitialLearningRate->\[Lambda]}];

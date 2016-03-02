(* ::Package:: *)

(*
   See Yann Le Cunn et al 1998:

http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

Network Architecture: Convolutional Network, see Figure 2 in above paper

They claim .9% test result

Our Results:
Iteration: 200
Training Loss: .011
Training Classification: .004%

Validation Loss: .111
Validation Classification: 1.8%

Looks like we may have slightly overtrained
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
      wl,TrainingImages[[1;;49000]],MNISTTrainingTargets[[1;;49000]],
      NNGrad,ClassificationLoss,
        {MaxEpoch->500000,
         ValidationInputs->TrainingImages[[49001;;50000]],
         ValidationTargets->MNISTTrainingTargets[[49001;;50000]],
         StepMonitor->NNCheckpoint["MNIST\\LeNet5"],
         LearningRate->\[Lambda]}];

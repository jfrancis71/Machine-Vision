(* ::Package:: *)

(*
   See Yann Le Cunn et al 1995:

http://yann.lecun.com/exdb/publis/pdf/lecun-95b.pdf

Network Architecture: Large Fully Connected Multi-Layer Neural Network
                      400-300-10
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX REWRITE RESULTS XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx
They claim 1.6% test result

Our Results:
Iteration: 3201
Training Loss: .010
Training Classification: 99.9%

Validation Loss: .090
Validation Classification: 97.3%
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxx
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/MNIST/MNISTData.m"


MNISTLeNet95Network={
   Adaptor2DTo1D[28],
   FullyConnected1DTo1DInit[28*28,300],Tanh,
   FullyConnected1DTo1DInit[300,10],
   Softmax};


wl=MNISTLeNet95Network;
TrainingHistory={};
ValidationHistory={};
\[Lambda]=.01;


TrainLeNet95:=MiniBatchGradientDescent[
      wl,TrainingImages,MNISTTrainingTargets,
      NNGrad,ClassificationLoss,
        {MaxEpoch->200,
         ValidationInputs->ValidationImages,
         ValidationTargets->MNISTValidationTargets,
         StepMonitor->NNCheckpoint["MNIST\\LeNet95"],
         InitialLearningRate->\[Lambda]}];

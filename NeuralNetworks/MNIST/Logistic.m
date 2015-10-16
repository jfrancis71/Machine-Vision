(* ::Package:: *)

(*
   Technique reviewed by Yann Le Cunn et al 1995:

http://yann.lecun.com/exdb/publis/pdf/lecun-95b.pdf

Network Architecture: Linear Network
                      784-10

They claim 8.4% test result (i.e. 91.6% correct)

Our Results:
Iteration: 
Training Loss: 
Training Classification: 

Validation Loss: 
Validation Classification: 


*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/MNIST/MNISTData.m"


SeedRandom[1234];
MNISTLinearNetwork={
   Adaptor2DTo1D[28],
   FullyConnected1DTo1DInit[784,10],
   Softmax};


wl=MNISTLinearNetwork;
TrainingHistory={};
ValidationHistory={};
\[Lambda]=.01;


TrainLogistic:=MiniBatchGradientDescent[
      wl,TrainingImages,MNISTTrainingTargets,
      NNGrad,ClassificationLoss,
        {MaxEpoch->200,
         ValidationInputs->ValidationImages,
         ValidationTargets->MNISTValidationTargets,
         StepMonitor->NNCheckpoint["MNIST\\Logistic"],
         InitialLearningRate->\[Lambda]}];

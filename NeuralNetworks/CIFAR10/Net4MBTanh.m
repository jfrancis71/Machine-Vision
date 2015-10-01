(* ::Package:: *)

(*

   CAFFE Ref:

   Differences in pooling arrangements

   Not completely faithful implementation.

   Epoch: 66
   Training Loss: .563
   Validation Loss: .943

   Validation Accuracy: 66.8%

*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/CIFAR10/CIFAR10Data.m"


SeedRandom[1234];
CIFARNet={
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[3,32,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[32,32,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[32,64,5],Tanh,
   MaxPoolingFilterBankToFilterBank,
   Adaptor3DTo1D[64,4,4],
   FullyConnected1DTo1DInit[64*4*4,64],Tanh,
   FullyConnected1DTo1DInit[64,10],
   Softmax
};


CIFAR10NetTrainingInputs=ColTrainingImages[[1;;4000]]*1.;
CIFAR10NetTrainingOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels];

CIFAR10NetValidationInputs=ColValidationImages[[All]]*1.;
CIFAR10NetValidationOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,ValidationLabels];


wl=CIFARNet;
TrainingHistory={};
ValidationHistory={};
\[Lambda]=.01;


TrainNet:=MiniBatchGradientDescent[
      wl,ColTrainingImages,CIFAR10NetTrainingOutputs,
      NNGrad,ClassificationLoss,
        {MaxEpoch->500000,
         ValidationInputs->ColValidationImages[[1;;500]],
         ValidationTargets->CIFAR10NetValidationOutputs[[1;;500]],
         UpdateFunction->NNCheckpoint["CIFAR10\\Net4MBTanh"],
         InitialLearningRate->\[Lambda]}];

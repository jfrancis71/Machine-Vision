(* ::Package:: *)

(*

   Ref: https://code.google.com/p/cuda-convnet/
   Ref: https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-18pct.cfg
   Ref: Alex Krizhevsky

   Not completely faithful implementation.
   He uses RELU and a contrast normalisation layer.

   Epoch: 48
   Training Loss: .746
   Validation Loss: 1.19

*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/CIFAR10/CIFAR10Data.m"


SeedRandom[1234];
CIFARNet3={
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[3,32,5],ReLU,
   MaxPoolingFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[32,32,5],ReLU,
   MaxPoolingFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[32,64,5],ReLU,
   MaxPoolingFilterBankToFilterBank,
   Adaptor3DTo1D[64,4,4],
   FullyConnected1DTo1DInit[64*4*4,10],
   Softmax
};


CIFAR10NetTrainingInputs=ColTrainingImages[[1;;4000]]*1.;
CIFAR10NetTrainingOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels];

CIFAR10NetValidationInputs=ColValidationImages[[All]]*1.;
CIFAR10NetValidationOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,ValidationLabels];


wl=CIFARNet3;
TrainingHistory={};
ValidationHistory={};
\[Lambda]=.01;


TrainNet3MBReLU:=MiniBatchGradientDescent[
      wl,ColTrainingImages,CIFAR10NetTrainingOutputs,
      NNGrad,ClassificationLoss,
        {MaxEpoch->500000,
         ValidationInputs->ColValidationImages[[1;;500]],
         ValidationTargets->CIFAR10NetValidationOutputs[[1;;500]],
         UpdateFunction->NNCheckpoint["CIFAR10//Net3MBReLU"],
         InitialLearningRate->\[Lambda]}];


CIFAR10Output[probs_]:=BarChart[probs[[Reverse[Ordering[probs,-3]]]],ChartLabels->WordLabels[[Reverse[Ordering[probs,-3]]]]];


CIFAR10Outputs[probs_,pics_]:=MapThread[{Image[#2,Interleaving->False],CIFAR10Output[#1]}&,{probs,pics}];

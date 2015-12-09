(* ::Package:: *)

(*

   Ref: https://code.google.com/p/cuda-convnet/
   https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-80sec.cfg
   Ref: Alex Krizhevsky

   Not completely faithful implementation.
   He uses RELU and has an extra 64 flat layer (64*4*4->64->10)
   Claims 74% classification performance

   Training Loss: .497
   Validation Loss: .787

   Classification Performance: 74.2%
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/CIFAR10/CIFAR10Data.m"


SeedRandom[1234];
CIFARNet5={
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[3,32,5],Tanh,
   MaxConvolveFilterBankToFilterBank,SubsampleFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[32,32,5],Tanh,
   MaxConvolveFilterBankToFilterBank,SubsampleFilterBankToFilterBank,
   PadFilterBank[2],ConvolveFilterBankToFilterBankInit[32,64,5],Tanh,
   MaxConvolveFilterBankToFilterBank,SubsampleFilterBankToFilterBank,
   Adaptor3DTo1D[64,4,4],
   FullyConnected1DTo1DInit[64*4*4,10],
   Softmax
};



CIFAR10NetTrainingOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels];

CIFAR10NetValidationInputs=ColValidationImages[[All]]*1.;
CIFAR10NetValidationOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,ValidationLabels];


wl=CIFARNet5;
TrainingHistory={};
ValidationHistory={};
\[Lambda]=.01;


TrainNet6MB:=MiniBatchGradientDescent[
      wl,ColTrainingImages[[1;;49000]],CIFAR10NetTrainingOutputs[[1;;49000]],
      NNGrad,ClassificationLoss,
        {MaxEpoch->500000,
         ValidationInputs->ColTrainingImages[[-1000;;-1]],
         ValidationTargets->CIFAR10NetTrainingOutputs[[-1000;;-1]],
         StepMonitor->NNCheckpoint["CIFAR10\\Net6MB"],
         Momentum->0.9,MomentumType->"Nesterov",
         InitialLearningRate->\[Lambda]}];


CIFAR10Output[probs_]:=BarChart[probs[[Reverse[Ordering[probs,-3]]]],ChartLabels->WordLabels[[Reverse[Ordering[probs,-3]]]]];


CIFAR10Outputs[probs_,pics_]:=MapThread[{Image[#2,Interleaving->False],CIFAR10Output[#1]}&,{probs,pics}];

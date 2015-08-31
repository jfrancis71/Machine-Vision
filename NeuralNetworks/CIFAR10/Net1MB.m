(* ::Package:: *)

(*
   Based on Andrej Karpathy. Not completely faithful implementation. (He uses extra convolution layer and ReLU)
   See: http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html

   Epoch: 27
   Training Loss: .848
   Validation Loss: 1.38
   Validation Classification Performance: 54.8%

*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/CIFAR10/CIFAR10Data.m"


SeedRandom[1234];
CIFARNet1={
   ConvolveFilterBankToFilterBank[Table[
      ConvolveFilterBankTo2D[0,{
         (Table[Random[],{5},{5}]-.5),
         (Table[Random[],{5},{5}]-.5),
         (Table[Random[],{5},{5}]-.5)}/(25*3.)],
      {f,1,32}]],Tanh,
   MaxPoolingFilterBankToFilterBank,
   ConvolveFilterBankToFilterBank[Table[
      ConvolveFilterBankTo2D[0,
         (Table[Random[],{32},{5},{5}]-.5)/(32*5*5)],
      {f,1,32}]],Tanh,
   Adaptor3DTo1D[32,10,10],
   FullyConnected1DTo1D[Table[.0,{10}],Table[Random[]-.5,{10},{32*10*10}]],
   Softmax
};


CIFAR10NetTrainingInputs=ColTrainingImages[[1;;4000]]*1.;
CIFAR10NetTrainingOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels];

CIFAR10NetValidationInputs=ColValidationImages[[All]]*1.;
CIFAR10NetValidationOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,ValidationLabels];


wl=CIFARNet;
TrainingHistory={};
ValidationHistory={};


TrainNet1MB:=MiniBatchGradientDescent[
      wl,ColTrainingImages,CIFAR10NetTrainingOutputs,
      NNGrad,ClassificationLoss,
        {MaxEpoch->500000,
         ValidationInputs->ColValidationImages[[1;;500]],
         ValidationTargets->CIFAR10NetValidationOutputs[[1;;500]],         
         UpdateFunction->WebMonitor["CIFAR10\\Net1MB"],
         InitialLearningRate->\[Lambda]}];


CIFAR10Output[probs_]:=BarChart[probs[[Reverse[Ordering[probs,-3]]]],ChartLabels->WordLabels[[Reverse[Ordering[probs,-3]]]]];


CIFAR10Outputs[probs_,pics_]:=MapThread[{Image[#2,Interleaving->False],CIFAR10Output[#1]}&,{probs,pics}];

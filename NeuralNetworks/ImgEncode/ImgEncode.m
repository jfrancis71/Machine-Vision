(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/CIFAR10/CIFAR10Data.m"


Images=Sqrt[Map[Reverse[Apply[Plus,#^2]]&,ColTrainingImages[[1;;4000]]]]/3.;


SeedRandom[1234];
Net1={
   FullyConnected1DTo1D[Table[Random[],{100}],Table[Random[]-.5,{100},{32*32}]],
   Tanh,
   FullyConnected1DTo1D[Table[Random[],{32*32}],Table[Random[]-.5,{32*32},{100}]]
};


wl=Net1;
TrainingHistory={};
ValidationHistory={};


Fl=Map[Flatten,Images];


Net1KTanhTrain:=(
   name="ImgEncode\\Net1Sz1KTanh";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   AdaptiveGradientDescent[
      wl,Fl[[1;;1000]],Fl[[1;;1000]],
      BatchGrad,RegressionLoss1D,
        {MaxLoop->500000,
         UpdateFunction->SkipWebMonitor[name],
         InitialLearningRate->\[Lambda]}];
)

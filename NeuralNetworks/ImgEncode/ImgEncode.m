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

Net2={
   FullyConnected1DTo1D[Table[-10,{100}]-.5,Table[(Random[]-.5),{100},{32*32}]/(1000*32*32)],
   Logistic,
   FullyConnected1DTo1D[Table[.2759,{32*32}],Table[Random[]-.5,{32*32},{100}]/1000000]
};


wl=Net2;
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
         UpdateFunction->SkipWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)


Net2Sz1KLogTrain:=(
   name="ImgEncode\\Net2Sz1KLog";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   AdaptiveGradientDescent[
      wl,Fl[[1;;1000]],Fl[[1;;1000]],
      BatchGrad,RegressionLoss1D,
        {MaxLoop->50000,
         UpdateFunction->SkipWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)

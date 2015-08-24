(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/CIFAR10/CIFAR10Data.m"


Images=Sqrt[Map[Reverse[Apply[Plus,#^2]]&,ColTrainingImages[[1;;4000]]]]/3.;


SparseRandom[]:=If[Random[]>.995,Random[],Random[]/1000]


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

Net3={
   FullyConnected1DTo1D[Table[Random[]-.5,{100}]-.5,Table[(Random[]-.5),{100},{32*32}]/(1000*32*32)],
   FullyConnected1DTo1D[Table[Random[]-.5,{32*32}],Table[Random[]-.5,{32*32},{100}]/1000000]
};

Net4={
   FullyConnected1DTo1D[Table[0.5,{100}],Table[SparseRandom[],{100},{32*32}]/(1000*32*32)],
   Logistic,
   FullyConnected1DTo1D[Table[Random[]-.5,{32*32}],Table[Random[]-.5,{32*32},{100}]/1000000]
};

Net5={
   FullyConnected1DTo1D[Table[0.5,{100}],Table[ReplacePart[ConstantArray[0.,{32*32}],1+RandomInteger[(32*32)-1]->1.],{100}]],
   Logistic,
   FullyConnected1DTo1D[Table[Random[]-.5,{32*32}],Table[Random[]-.5,{32*32},{100}]/1000000]
};

Net6={
   FullyConnected1DTo1D[Table[0.,{100}],Table[(Random[]-.5)*Sqrt[2/(32*32)],{100},{32*32}]],
   Tanh,
   FullyConnected1DTo1D[Table[0.,{32*32}],Table[(Random[]-.5)*Sqrt[2/100],{32*32},{100}]]
};

Net7={
   FullyConnected1DTo1D[Table[0.,{100}],Table[(Random[]-.5)*Sqrt[2/(32*32)],{100},{32*32}]],
   Tanh,
   FullyConnected1DTo1D[Table[0.,{32*32}],Table[0.,{32*32},{100}]]
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
         UpdateFunction->CheckpointWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)


Net3Sz1KTrain:=(
   name="ImgEncode\\Net3Sz1K";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   AdaptiveGradientDescent[
      wl,Fl[[1;;1000]],Fl[[1;;1000]],
      BatchGrad,RegressionLoss1D,
        {MaxLoop->50000,
         UpdateFunction->CheckpointWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)


Net4Sz1KTrain:=(
   name="ImgEncode\\Net4Sz1K";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   AdaptiveGradientDescent[
      wl,Fl[[1;;1000]],Fl[[1;;1000]],
      BatchGrad,RegressionLoss1D,
        {MaxLoop->500000,
         UpdateFunction->CheckpointWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)


NetDenoise1Train:=(
   name="ImgEncode\\NetDenoise1";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   AdaptiveGradientDescent[
      wl,Fl[[1;;1000]]+noise,Fl[[1;;1000]],
      BatchGrad,RegressionLoss1D,
        {MaxLoop->500000,
         UpdateFunction->CheckpointWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)


Net5Train:=(
   name="ImgEncode\\Net5";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   AdaptiveGradientDescent[
      wl,Fl[[1;;1000]]+noise,Fl[[1;;1000]],
      BatchGrad,RegressionLoss1D,
        {MaxLoop->500000,
         UpdateFunction->CheckpointWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)

Net6Train:=(
   name="ImgEncode\\Net6";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   GradientDescent[
      wl,Fl[[1;;100]]+noise[[1;;100]],Fl[[1;;100]],
      Grad,RegressionLoss1D,
        {MaxLoop->500000,
         UpdateFunction->CheckpointWebMonitor[name,500],
         InitialLearningRate->\[Lambda]}];
)


Net6Train:=(
   name="ImgEncode\\Net6";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   MBGDWithNoise[
      wl,Fl[[1;;4000]],Fl[[1;;4000]],
      Grad,RegressionLoss1D,
        {MaxLoop->500000,
         UpdateFunction->CheckpointWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)


TiedGrad[wl_,inputs_,targets_,lossF_]:=(
   t1=ReplacePart[BatchGrad[ReplacePart[wl,{3,2}->Transpose[wl[[1,2]]]],inputs,targets,RegressionLoss1D],{3,2}->Table[0.,{32*32},{100}]];
   tt2=Transpose[BatchGrad[ReplacePart[wl,{3,2}->Transpose[wl[[1,2]]]],inputs,targets,RegressionLoss1D][[3,2]]];
   t1[[1,2]]+=tt2;
   t1)


TiedRegressionLoss1D[wl_,inputs_,targets_]:=
   RegressionLoss1D[ReplacePart[wl,{3,2}->Transpose[wl[[1,2]]]],inputs,targets]


Net7Train:=(
   name="ImgEncode\\Net7";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   AdaptiveGradientDescent[
      wl,Fl[[1;;1000]],Fl[[1;;1000]],
      TiedGrad,TiedRegressionLoss1D,
        {MaxLoop->500000,
         UpdateFunction->CheckpointWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)

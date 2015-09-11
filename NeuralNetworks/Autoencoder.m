(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


AddNoise[inputs_]:=(
   tmp1=UnitStep[RandomReal[{0,1},inputs//Dimensions]-.667];
   (inputs*(1-tmp1)+tmp1*RandomReal[{0,1},inputs//Dimensions])
)


NoisyTiedGrad[wl_,inputs_,targets_,lossF_,options_:{}]:=
   TiedGrad[wl,AddNoise[inputs],targets,lossF,options];


UnTieLoss[TiedCrossEntropyLoss]:=CrossEntropyLoss


UnTieLoss[TiedRegressionLoss1D]:=RegressionLoss1D


TieNet[net_]:=ReplacePart[net,{3,2}->Transpose[net[[1,2]]]]


TiedGrad[wl_,inputs_,targets_,lossF_,options_:{}]:=(
   t1=ReplacePart[NNGrad[TieNet[wl],inputs,targets,UnTieLoss[lossF],options],{3,2}->wl[[3,2]]*0.0];
   t2=Transpose[NNGrad[TieNet[wl],inputs,targets,UnTieLoss[lossF],options][[3,2]]];
   t3=t1;
   t4=t3;
   t4[[1,2]]+=t2;
   t4)


CrossEntropyLoss[parameters_,inputs_,targets_]:=
   -Total[targets*Log[ForwardPropogate[inputs,parameters]]+(1-targets)*Log[1-ForwardPropogate[inputs,parameters]],2]/Length[inputs]


DeltaLoss[CrossEntropyLoss,outputs_,targets_]:=-((-(1-targets)/(1-outputs)) + (targets/outputs))/Length[outputs];


TiedCrossEntropyLoss[wl_,inputs_,targets_]:=
   CrossEntropyLoss[TieNet[wl],inputs,targets]


TiedRegressionLoss1D[wl_,inputs_,targets_]:=
   RegressionLoss1D[TieNet[wl],inputs,targets]


TieLoss[CrossEntropyLoss]:=TiedCrossEntropyLoss


TieLoss[RegressionLoss1D]:=TiedRegressionLoss1D


TieLoss[_]:=AbortAssert[0,"TieLoss: Unrecognised loss function"];


TrainAutoencoder[in_Integer,out_Integer,data_,lossF_]:=(
   SeedRandom[1234];
   net={
      FullyConnected1DTo1DInit[in,out],
      Logistic,
      FullyConnected1DTo1DInit[out,in],
      Logistic};
   MiniBatchGradientDescent[
      net,data,data,
      NoisyTiedGrad,TieLoss[lossF],
        {MaxEpoch->50,
        UpdateFunction->ScreenMonitor,
         InitialLearningRate->.1}];
   {wl[[1;;2]],TieNet[wl][[3;;4]]})

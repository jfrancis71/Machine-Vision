(* ::Package:: *)

(*
   Extracting and composing robust features with denoising autoencoders, Vincent et al (Feb 2008)
   Ref: http://www.iro.umontreal.ca/~vincentp/Publications/denoising_autoencoders_tr1316.pdf
*)


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


TiedCrossEntropyLoss[wl_,inputs_,targets_]:=
   CrossEntropyLoss[TieNet[wl],inputs,targets]


TiedRegressionLoss1D[wl_,inputs_,targets_]:=
   RegressionLoss1D[TieNet[wl],inputs,targets]


TieLoss[CrossEntropyLoss]:=TiedCrossEntropyLoss


TieLoss[RegressionLoss1D]:=TiedRegressionLoss1D


TieLoss[_]:=AbortAssert[0,"TieLoss: Unrecognised loss function"];


Options[TrainAutoencoder] = { MaxIterations -> 2000 };


TrainAutoencoder[in_Integer,out_Integer,data_,lossF_,opts___]:=(
   SeedRandom[1234];
   net={
      FullyConnected1DTo1DInit[in,out],
      Logistic,
      FullyConnected1DTo1DInit[out,in],
      Logistic};
   MiniBatchGradientDescent[
      net,data,data,
      NoisyTiedGrad,TieLoss[lossF],
        {MaxEpoch->(MaxIterations/.{opts}/.Options[TrainAutoencoder]),
        UpdateFunction->ScreenMonitor,
         InitialLearningRate->.1}];
   {wl[[1;;2]],TieNet[wl][[3;;4]]})


TrainStackedAutoencoder[dat_]:=(
   TrainingHistory={};

   AbortAssert[MatrixQ[dat],"TrainStackedAutoencoder::Data must be flat."];

   layer1Dat=dat;
   {encoder1,decoder1}=TrainAutoencoder[layer1Dat[[1]]//Length,500,layer1Dat,CrossEntropyLoss];

   layer2Dat=ForwardPropogate[layer1Dat,encoder1];
   {encoder2,decoder2}=TrainAutoencoder[500,250,layer2Dat,RegressionLoss1D];

   layer3Dat=ForwardPropogate[layer2Dat,encoder2];
   {encoder3,decoder3}=TrainAutoencoder[250,125,layer3Dat,RegressionLoss1D];

   layer4Dat=ForwardPropogate[layer3Dat,encoder3];
   {encoder4,decoder4}=TrainAutoencoder[125,64,layer4Dat,RegressionLoss1D];

   {encoder,decoder}={Flatten[{encoder1,encoder2,encoder3,encoder4}],Flatten[{decoder1,decoder2,decoder3,decoder4}//Reverse]};
)

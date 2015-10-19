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


Options[TrainAutoencoder] = {};


TrainAutoencoder[in_Integer,out_Integer,data_,lossF_,opts:OptionsPattern[]]:=(
   SeedRandom[1234];
   net={
      FullyConnected1DTo1DInit[in,out],
      Logistic,
      FullyConnected1DTo1DInit[out,in],
      Logistic};
   MiniBatchGradientDescent[
      net,data,data,
      NoisyTiedGrad,TieLoss[lossF],
        opts];
   {wl[[1;;2]],TieNet[wl][[3;;4]]})


PreTrainStackedAutoencoder[dat_?MatrixQ,layers_?VectorQ,opts:OptionsPattern[]]:=
   If[Length[layers]>=2,
      Module[{
         codec=TrainAutoencoder[First[layers],First[Rest[layers]],dat,RegressionLoss1D,opts]},
(* Slightly odd nesting because of dependancy of second definition on first *)
         Module[{
            codecs=PreTrainStackedAutoencoder[ForwardPropogate[dat,codec[[1]]],Rest[layers],opts]},
            {Join[codec[[1]],codecs[[1]]],Join[codecs[[2]],codec[[2]]]}]],
   {{},{}}]


TrainStackedAutoencoder[dat_,opts:OptionsPattern[]]:=(
   TrainingHistory={};

   AbortAssert[MatrixQ[dat],"TrainStackedAutoencoder::Data must be flat."];

   layer1Dat=dat;
   {encoder1,decoder1}=TrainAutoencoder[layer1Dat[[1]]//Length,500,layer1Dat,CrossEntropyLoss,opts];

   layer2Dat=ForwardPropogate[layer1Dat,encoder1];
   {encoder2,decoder2}=TrainAutoencoder[500,250,layer2Dat,RegressionLoss1D,opts];

   layer3Dat=ForwardPropogate[layer2Dat,encoder2];
   {encoder3,decoder3}=TrainAutoencoder[250,125,layer3Dat,RegressionLoss1D,opts];

   layer4Dat=ForwardPropogate[layer3Dat,encoder3];
   {encoder4,decoder4}=TrainAutoencoder[125,64,layer4Dat,RegressionLoss1D,opts];

   {encoder,decoder}={Flatten[{encoder1,encoder2,encoder3,encoder4}],Flatten[{decoder1,decoder2,decoder3,decoder4}//Reverse]};
)

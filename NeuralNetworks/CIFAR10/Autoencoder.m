(* ::Package:: *)

(* Learns a basic denoising autoencoder trained on CIFAR-10
   Does replicate learning features observed in the literature even if it is not exactly Gabor like.
   It does seem to do a pretty good job of denoising a corrupted input, albeit output is somewhat blurred.
   Subjectively the noise is quite severe and I would have a hard time reconstructing the original image.
   Loss around 8.4 after 1,600 epochs
   which presumably corresponds to mean pixel error of around 10%

   Ref: http://deeplearning.net/tutorial/dA.html
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/Autoencoder.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/CIFAR10/CIFAR10Data.m"


SeedRandom[1234];

Net1={
   FullyConnected1DTo1D[Table[0.5,{500}],Table[(Random[]-.5)*4*Sqrt[6/(32*32+500)],{500},{32*32}]],
   Logistic,
   FullyConnected1DTo1D[Table[0.,{32*32}],Table[0.,{32*32},{500}]],
   Logistic
};


Images=.2*ColTrainingImages[[All,1]]+.73*ColTrainingImages[[All,2]]+.07*ColTrainingImages[[All,3]];


Fl=Map[Flatten,Images];


(* Produced Autoencoder.wdx *)  
Train:=MiniBatchGradientDescent[
      wl,Fl[[1;;10000]],Fl[[1;;10000]],
      NoisyTiedGrad,TiedRegressionLoss1D,
        {MaxEpoch->500000,
         ValidationInputs->Fl[[4001;;4500]],
         ValidationTargets->Fl[[4001;;4500]],         
         InitialLearningRate->.01}];


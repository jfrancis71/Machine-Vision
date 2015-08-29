(* ::Package:: *)

(*
   Ref: CIFAR 10 - 
      Learning Multiple Layers Of Features From Tiny Images
      Krizhevsky - 2009
   http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


dat=Join[
   BinaryReadList["C:\\Users\\Julian\\ImageDataSetsPublic\\CIFAR-10\\data_batch_1.bin"](*,
   BinaryReadList["C:\\Users\\Julian\\ImageDataSetsPublic\\CIFAR-10\\data_batch_2.bin"],
   BinaryReadList["C:\\Users\\Julian\\ImageDataSetsPublic\\CIFAR-10\\data_batch_3.bin"],
   BinaryReadList["C:\\Users\\Julian\\ImageDataSetsPublic\\CIFAR-10\\data_batch_4.bin"],
   BinaryReadList["C:\\Users\\Julian\\ImageDataSetsPublic\\CIFAR-10\\data_batch_5.bin"]*)];


records=Partition[dat,3073];


reorder=records;(*Actually decided to retain original order to do with reproducibility *)


labels=Map[First,reorder];


picVector=Map[Rest,reorder];


TrainingImages=Table[((Partition[Sqrt[Total[Partition[picVector[[l]],1024]^2]],32]//Reverse)//N)/443,
{l,1,10000}];


TrainingLabels=labels[[1;;4500]];


ValidationImages=Table[((Partition[Sqrt[Total[Partition[picVector[[l]],1024]^2]],32]//Reverse)//N)/443,
{l,4501,5000}];


ValidationLabels=labels[[4501;;5000]];


ColTrainingImages=Table[Map[Partition[#,32]&,Partition[picVector[[l]],1024]]/256.,{l,1,10000}];


ColValidationImages=Table[Map[Partition[#,32]&,Partition[picVector[[l]],1024]]/256.,{l,4501,5000}];


WordLabels={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};

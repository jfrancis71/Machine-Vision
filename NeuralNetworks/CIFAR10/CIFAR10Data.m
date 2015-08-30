(* ::Package:: *)

(*
   Ref: CIFAR 10 - 
      Learning Multiple Layers Of Features From Tiny Images
      Krizhevsky - 2009
   http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


WordLabels={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};


CIFARReadFile[filename_]:={
Map[First,Partition[BinaryReadList[filename],3073]],
Map[Function[{flat},Partition[flat,32]],Map[Partition[#,1024]&,Map[Rest,Partition[BinaryReadList[filename],3073]]],{2}]/256.
}


CIFARFiles={"c:\\Users\\Julian\\ImageDataSetsPublic\\CIFAR-10\\data_batch_1.bin",
   "c:\\Users\\Julian\\ImageDataSetsPublic\\CIFAR-10\\data_batch_2.bin",
   "c:\\Users\\Julian\\ImageDataSetsPublic\\CIFAR-10\\data_batch_3.bin",
   "c:\\Users\\Julian\\ImageDataSetsPublic\\CIFAR-10\\data_batch_4.bin"
};


TrainingLabels=Flatten[
   Map[CIFARReadFile,CIFARFiles]
   [[All,1]]];


ColTrainingImages=Flatten[
   Map[CIFARReadFile,
   CIFARFiles]
   [[All,2]],1];


ValidationLabels=CIFARReadFile["c:\\Users\\Julian\\ImageDataSetsPublic\\CIFAR-10\\data_batch_5.bin"][[1]];


ColValidationImages=CIFARReadFile["c:\\Users\\Julian\\ImageDataSetsPublic\\CIFAR-10\\data_batch_5.bin"][[2]];

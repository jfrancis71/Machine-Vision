(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/Legology/EV3Library.m"


EV3State="STOP";


EV3Forward[]:=(EV3StepMotor[EV3PortB];EV3StepMotor[EV3PortC];)


EV3Stop[]:=(EV3StopMotor[EV3PortB];EV3StopMotor[EV3PortC];)


EV3Left[]:=(EV3StepMotor[EV3PortC];EV3StopMotor[EV3PortB];)


EV3Right[]:=(EV3StepMotor[EV3PortB];EV3StopMotor[EV3PortC];)


op


TrainingImages={};TrainingCommands={};


hist={}


RemoteControlCarTraining[image_]:=( 
   prev=SessionTime[];
   out=DispImage[image];
   state=ControllerState[{"B1","X"}];
   If[state[[1]],
      AppendTo[TrainingImages,image];
      If[Abs[state[[2]]]<.5,
         EV3Forward[];AppendTo[TrainingCommands,0],
         If[state[[2]]<-.5,EV3Left[];AppendTo[TrainingCommands,-1],EV3Right[];AppendTo[TrainingCommands,+1]]]
      ,EV3Stop[]];
   Pause[.2];
   cut=SessionTime[];
   AppendTo[hist,{cut-old,cut-prev}];
   out
)


RemoteControlCarDrive[image_]:=(
   out=DispImage[image];
   netout=ForwardPropogate[{image},wl][[1]];
   cmd=Position[netout,Max[netout]][[1,1]]-2;
   Switch[cmd,
      -1,EV3Left[],
      0,EV3Forward[],
      +1,EV3Right[]
];
Pause[0.2];
out)

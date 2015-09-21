(* ::Package:: *)

(* ::Subtitle:: *)
(*WallFinder*)


(* ::Text:: *)
(*Program will drive EV3 up to a wall and stop. Note it will back off if it senses something in front of it.*)
(**)
(*Procedure:*)
(*Call WallFinder[]*)
(**)


(* ::Input:: *)
(*SetDirectory["c:/users/julian/secure/legotastic/"]*)


(* ::Input:: *)
(*<<EV3Library`*)


(* ::Input:: *)
(*tmp = SerialOpen["COM3"];If[tmp===$Failed,,mybrick=tmp]*)


(* ::Input:: *)
(*SerialClose[mybrick]*)


(* ::Input:: *)
(*WallFinder[]:=( *)
(*targetDistance=15;*)
(*blackbox={};*)
(*While[True,*)
(*currentDistance=EV3ReadSensor[];*)
(*pow=Round[(currentDistance-targetDistance)*4.0];*)
(*AppendTo[blackbox,{currentDistance,pow}];*)
(*EV3StartMotor[EV3PortB,pow];*)
(*EV3StartMotor[EV3PortC,pow];*)
(*Pause[1]*)
(*]*)
(*)*)


(* ::Input:: *)
(*WallFinder[]*)


(* ::Input:: *)
(*pow*)


(* ::Input:: *)
(*(currentDistance-targetDistance)*0.3*)


(* ::Input:: *)
(*blackbox*)


(* ::Input:: *)
(*Dynamic[{pow,currentDistance}]*)


(* ::Input:: *)
(*EV3StopMotor[EV3PortB];*)
(*EV3StopMotor[EV3PortC];*)

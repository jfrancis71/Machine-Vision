(* ::Package:: *)

(* ::Subtitle:: *)
(*EV3 Control Library*)
(**)
(**)


Author: Julian W Francis                  julian.w.francis@gmail.com


<< SerialIO`

Pause[10]; (*Deal with race condition in one of the dependant libraries *)
tmp = SerialOpen["COM3"];If[tmp===$Failed,,mybrick=tmp];

EV3PortA = 1;
EV3PortB = 2;
EV3PortC = 4;
EV3PortD = 8;

EV3InputPortB=17;
EV3InputPortC=18;

(*actually speed*)
EV3StartMotor[motor_,power_] := 
(
moderatedPower=If[Abs[power]>100,100*Sign[power],power];
   telegram = {16^^0D, 16^^00,(*LENGTH*)
     16^^01, 16^^00 (*Response Sequence*),
     16^^80,(*NoReply*)
     16^^00, 16^^00,
     16^^A5, (*OutputSpeed*)
     16^^00,(*LAYER, whatever that means*)
     motor, (*Port*)
     16^^81,
     If[moderatedPower>=0,moderatedPower,256-Abs[moderatedPower]], (*Power*)
     16^^A6,
     16^^00, motor};
   message = Map[FromCharacterCode, telegram];
   Map[SerialWrite[mybrick, #] &, message]
   );

EV3StartMotor[motor_] := EV3StartMotor( motor, 16^^10 );


EV3StepMotor[motor_]:=(
   telegram = {16^^12, 16^^00,(*LENGTH*)
     16^^01, 16^^00 (*Response Sequence*),
     16^^80,(*NoReply*)
     16^^00, 16^^00,
     16^^AE, (*StepSpeed OP*)
     16^^00,(*LAYER, whatever that means*)
     motor, (*Port*)
     16^^81,20,(*Power*)
     00,
     16^^82,10,00,
     16^^82,16^^00,00,
     01};
   message = Map[FromCharacterCode, telegram];
   Map[SerialWrite[mybrick, #] &, message]
)


EV3StopMotor[motor_] := (
   telegram = {16^^09, 16^^00,(*LENGTH*)
     16^^01, 16^^00 (*Response Sequence*),
     16^^80,(*NoReply*)
     16^^00, 16^^00,
     16^^A3, (*OutputStop*)
     16^^00,(*LAYER, whatever that means*)
     motor, (*Port*)
     16^^00 (*Break*)
     };
   message = Map[FromCharacterCode, telegram];
   Map[SerialWrite[mybrick, #] &, message]
   );
 
EV3ReadSensor[port_:00] := (
   telegram = {
     16^^0B, 00,
     01, 00 (*Response Sequence*),
     00 (*Reply*),
     01, 00,
     16^^9A (*ReadSensor*),
     00, port, 00, 00, 16^^60};
   message = Map[FromCharacterCode, telegram];
   Map[SerialWrite[mybrick, #] &, message];
   reply = SerialRead[mybrick];
   ToCharacterCode[reply] // Last
   );

(*Make sure you use the EV3InputPortX as the argument *)
EV3ReadTachoSensor[port_:00] := (
reply={};
   telegram = {
     16^^0B, 00,
     01, 00 (*Response Sequence*),
     00 (*Reply*),
     04, 00,
     16^^9D (*ReadSensor*),
     00, port(*port*),00,00,16^^60 };
   message = Map[FromCharacterCode, telegram];
   Map[SerialWrite[mybrick, #] &, message];
   reply = SerialRead[mybrick];
   ToCharacterCode[reply][[-4]];
   ImportString[StringTake[reply,{6,9}],"Real32"]//First
   );


EV3ReadColourSensor[]:= (
defaultPort:=00;
   telegram = {
     16^^0B, 00,
     01, 00, 00, 01, 00, 16^^9A, 00, defaultPort, 00, 02, 16^^60};
   message = Map[FromCharacterCode, telegram];
   Map[SerialWrite[mybrick, #] &, message];
   reply = SerialRead[mybrick];
   Switch[ToCharacterCode[reply] // Last,
		12,"Black",(* 1 *)
		75,"White", (* 6 *)
		62,"Red", (* 5 *)
		0,"Transparent", (* 0 *)
		87,"Brown",(* 7 *)
		50,"Yellow", (* 4 *)
		37,"Green",(* 3 *)
		25,"Blue",(* 2 *)
		_,"ERROR"]
   );

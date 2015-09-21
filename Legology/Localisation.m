(* ::Package:: *)

(* ::Input:: *)
(*<<c:/users/julian/secure/legotastic/EV3Library`*)


(* ::Input:: *)
(*(* Control Logic *)*)


(* ::Input:: *)
(*map=ImageData[*)
(*ImageResize[Rasterize[Framed[Graphics[{Yellow,CountryData[#,"SchematicPolygon"]&/@CountryData[]},Background->Blue],Background->Red,FrameMargins->20]],300]];*)


(* ::Input:: *)
(*DiscreteGauss[x_,y_,d_]:=If[d==0&&x>=1&&y>=1&&x<=11&&y<=11,*)
(*GaussianMatrix[{5,0.5}][[Round[y],Round[x]]],0*)
(*]//N*)


(* ::Input:: *)
(*RotGauss[x_,y_,d_]:=If[d==0&&x>=1&&y>=1&&x<=81&&y<=81,*)
(*mat[[Round[y],Round[x]]],0*)
(*]//N*)


(* ::Input:: *)
(*updateState[currentState_,action_]:=Table[*)
(*Sum[*)
(*DiscreteGauss[*)
(*6+xs+(24.0/17.0)*Cos[\[Pi]/2 + dd-1] - xd,*)
(*6+ys+(24.0/17.0)*Sin[\[Pi]/2 + dd-1] -yd,*)
(*dd-ds]*currentState[[ds+1,ys,xs]]*)
(*,{ys,1,24},{xs,1,50},{ds,0,11}]*)
(*,{yd,1,24},{xd,1,50},{dd,0,11}];*)


(* ::Input:: *)
(*updateState[currentState_,action_]:=Table[*)
(*Sum[*)
(*DiscreteGauss[*)
(*6+xs+1 * Cos[\[Pi]/2 + (dd-1)*2\[Pi]/12] - xd,*)
(*6+ys+1 * Sin[\[Pi]/2 + (dd-1)*2\[Pi]/12] - yd,*)
(*dd-ds]*)
(**currentState[[ds,ys,xs]]*)
(*,{ys,1,24},{xs,1,50},{ds,1,12}]*)
(*,{yd,1,24},{xd,1,50},{dd,1,12}];*)


(* ::Input:: *)
(*stateKernel=RotateLeft[ArrayPad[GaussianMatrix[{5,0.5}],1]];*)


(* ::Input:: *)
(*stateKernels=Table[*)
(*DiscreteGauss[*)
(*6+x+1 * Cos[\[Pi]/2 + (d-1)*2\[Pi]/12],*)
(*6+y+1 * Sin[\[Pi]/2 + (d-1)*2\[Pi]/12],*)
(*0]*)
(*,{d,1,12},{y,-5,5},{x,-5,5}];*)
(*stateKernels=Map[#/Total[#,2]&,stateKernels];*)


(* ::Input:: *)
(*ArrayInBounds[matrix_,y_,x_]:=If[y>=1&&y<=Length[matrix]&&x>=1&&x<=Length[matrix[[1]]],matrix[[y,x]],0]*)


(* ::Input:: *)
(*ArrayInBounds[matrix_,y_,x_]:=If[y>=1&&y<=144&&x>=1&&x<=300,matrix[[y,x]],0]*)


(* ::Input:: *)
(*updateState[currentState_,action_ (* in radians *)]:=*)
(*If[action==-1,*)
(*Transpose[Map[ListConvolve[{0.01,0.98,0.01},#,2]&,Transpose[*)
(*MapThread[ListConvolve[*)
(*#1,*)
(*#2,{6,6},0]&,{stateKernels,currentState}]*)
(*,{3,2,1}],{2}],{3,2,1}] *)
(*(*MapThread[ListConvolve[*)
(*#1,*)
(*#2,{6,6},0]&,{stateKernels,currentState}]*)*)
(*,*)
(*(**)
(*t=Table[*)
(*ArrayInBounds[currentState[[d+1]],*)
(*Round[yd - 15 Cos[d*2 \[Pi]/12] + 15 Cos[d*2 \[Pi]/12 + action]],*)
(*Round[xd + 15 Sin[d*2 \[Pi]/12] - 15 Sin[d*2 \[Pi]/12 + action]]]*)
(* ,{d,0,11},{yd,1,144},{xd,1,300}];*)
(**)*)
(*coords=Table[*)
(*{d,*)
(*Round[yd - 15 Cos[d*2 \[Pi]/12.00] + 15 Cos[d*2 \[Pi]/12.0 + action]],*)
(*Round[xd + 15 Sin[d*2 \[Pi]/12.0] - 15 Sin[d*2 \[Pi]/12.0 + action]]}*)
(* ,{d,0,11},{yd,1,144},{xd,1,300}];*)
(*z1=Map[ArrayInBounds[currentState[[1+#[[1]]]],#[[2]],#[[3]]]&,coords,{3}];*)
(*z2=RotateRight[*)
(*Map[ListConvolve[GaussianMatrix[{20,5}],#,{21,21},0]&,z1],Round[12 action/(2 \[Pi])]];*)
(*Transpose[Map[ListConvolve[{0.1,0.8,0.1},#,1]&,Transpose[z2,{3,2,1}],{2}],{3,2,1}]*)
(*]*)


(* ::Input:: *)
(*Danger[y_,x_]:=If[y<=15||y>=144-15||x<=15||x>=300-15,True,False]*)


(* ::Input:: *)
(*Land[y_,x_]:=If[map[[y,x,3]]<0.2&&map[[y,x,2]]>0.8,True,False]*)


(* ::Input:: *)
(*Sea[y_,x_]:=Not[Danger[y,x]||Land[y,x]]*)


(* ::Input:: *)
(*updateObservation[currentState_,observation_]:=*)
(*Table[*)
(*currentState[[d,y,x]]*If[Sea[y,x],If[observation=="Blue",0.9,0.1],*)
(*If[Land[y,x],If[observation=="Yellow"||observation=="White",0.9,0.1],0.5]],*)
(*{d,1,12},{y,1,144},{x,1,300}]/*)
(*Sum[*)
(*currentState[[d,y,x]]*If[Sea[y,x],If[observation=="Blue",0.9,0.1],*)
(*If[Land[y,x],If[observation=="Yellow"||observation=="White",0.9,0.1],0.5]],*)
(*{y,1,144},{x,1,300},{d,1,12}]*)


(* ::Input:: *)
(*updateObservation[currentState_,observation_]:=( *)
(*obsMap=Table[*)
(*If[Sea[y,x],If[observation=="Blue",0.9,0.1],*)
(*If[Land[y,x],If[observation=="Yellow"||observation=="White",0.9,0.1],*)
(*If[Danger[y,x],If[observation=="Red",0.9,0.1],*)
(*0.5]]],*)
(*{y,1,144},{x,1,300}];*)
(*Map[*)
(*#*obsMap*)
(*&*)
(*,currentState]/Total[Map[*)
(*#*obsMap*)
(*&*)
(*,currentState],3]*)
(*)*)


(* ::Input:: *)
(*updateObservation[currentState_,observation_]:=( *)
(*obsMap=Table[*)
(*If[Sea[y,x],If[observation=="Blue",0.999,0.001],*)
(*If[Land[y,x],If[observation=="Yellow"||observation=="White",0.999,0.001],*)
(*If[Danger[y,x],If[observation=="Red",0.999,0.001],*)
(*0.5]]],*)
(*{y,1,144},{x,1,300}];*)
(*Map[*)
(*#*obsMap*)
(*&*)
(*,currentState]/Total[Map[*)
(*#*obsMap*)
(*&*)
(*,currentState],3]*)
(*)*)


(* ::Input:: *)
(*confKernel=ConstantArray[1,{12,21,21}];*)


(* ::Input:: *)
(*Process[]:=( *)
(*colour = EV3ReadColourSensor[];*)
(*currentState1=updateObservation[currentState,colour];*)
(**)
(*If[colour!="Red",*)
(*(* Go Forwards *)*)
(*EV3StepMotor[EV3PortB];*)
(*EV3StepMotor[EV3PortC];*)
(*action=-1;*)
(*Pause[0.1],*)
(*(* Spin *)*)
(*(*EV3StartMotor[EV3PortB,16];*)
(*EV3StartMotor[EV3PortC,-16];*)
(*f = Random[]*360;*)
(*Pause[f/12];*)
(*action=2 \[Pi] f/360*)iter=0*)
(*];*)
(*(**)
(*EV3StopMotor[EV3PortC];*)
(*EV3StopMotor[EV3PortB];*)*)
(*currentState2 = updateState[currentState1,action];*)
(*iter--;*)
(*If[record,AppendTo[blackbox,{colour,action,currentState2}]];*)
(*currentState=currentState2;*)
(*confMap=ListConvolve[confKernel,currentState2,{6,6},0];*)
(*confidence=Max[confMap];*)
(*)*)


(* ::Input:: *)
(*initialState=Table[1/(144*300*12),{d,1,12},{y,1,144},{x,1,300}];iter=15;blackbox={};*)


(* ::Input:: *)
(**)
(*Main[iteration_:100]:=*)
(*(currentState=initialState;*)
(*blackbox={};*)
(*iter=iteration;*)
(*EV3ReadColourSensor[];*)
(*While[iter>0,Process[]*)
(*])*)
(**)


(* ::Input:: *)
(*currentState=initialState;*)


(* ::Input:: *)
(*tmpGr=Graphics[{Transparent,EdgeForm[Green],Map[{200,80}+#&,CountryData[#,"Polygon"]&/@CountryData[],{4}]},Background->Transparent];*)


(* ::Input:: *)
(*DisplayProbMap[state_]:=*)
(*Show[ImageResize[Image[Total[state]],400]//ImageAdjust,tmpGr]*)


(* ::Input:: *)
(*rastMap=ColorCombine[{ImageApply[0&,Image[map]],EdgeDetect[Image[map]],ImageApply[0&,Image[map]]}];*)


(* ::Input:: *)
(*DisplayProbMap[state_]:=Show[ImageAdd[rastMap,Image[Total[state]]//ImageAdjust],ImageSize->Large]*)


(* ::Input:: *)
(*GUI[]:=Column[{*)
(*Dynamic[{colour,iter,action,confidence}],*)
(*Dynamic[DisplayProbMap[currentState]]*)
(*}]*)


(* ::Input:: *)
(*CalibrateTurn[]:=( *)
(*EV3StartMotor[EV3PortB,16];*)
(*EV3StartMotor[EV3PortC,-16];*)
(*f = 180;*)
(*Pause[f/12];*)
(*action=2 \[Pi] f/360;*)
(*EV3StopMotor[EV3PortC];*)
(*EV3StopMotor[EV3PortB];*)
(*)*)


(* ::Input:: *)
(*CalibrateMotion[]:=( *)
(*iter=144;*)
(*While[iter>=0,*)
(*EV3StepMotor[EV3PortB];*)
(*EV3StepMotor[EV3PortC];*)
(*Pause[0.1];(**)
(*EV3StopMotor[EV3PortC];*)
(*EV3StopMotor[EV3PortB];*)*)
(*Pause[0.5];*)
(*iter--;*)
(*]*)
(*)*)

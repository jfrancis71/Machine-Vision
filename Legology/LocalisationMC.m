(* ::Package:: *)

(* ::Subtitle:: *)
(*Monte Carlo Localisation*)


(* ::Text:: *)
(*Based on course notes by Andrew Davison (@ Imperial)*)


(* ::Input:: *)
(*NoParticles=20000;*)


(* ::Input:: *)
(*map=ImageData[*)
(*ImageResize[Rasterize[Framed[Graphics[{Yellow,CountryData[#,"SchematicPolygon"]&/@CountryData[]},Background->Blue],Background->Red,FrameMargins->20]],300]];*)


(* ::Input:: *)
(*Danger[y_,x_]:=If[y<=15||y>=144-15||x<=15||x>=300-15,True,False]*)


(* ::Input:: *)
(*Land[y_,x_]:=If[map[[y,x,3]]<0.2&&map[[y,x,2]]>0.8,True,False]*)


(* ::Input:: *)
(*Sea[y_,x_]:=Not[Danger[y,x]||Land[y,x]]*)


(* ::Input:: *)
(*initialParticles=Table[{1.0/NoParticles,RandomInteger[{1,300}]+0.0,RandomInteger[{1,144}]+0.0,Random[]*2*\[Pi]},{NoParticles}];*)


(* ::Input:: *)
(*(* Fix angle wraparound? *)*)
(*updateMotion[particle_,action_]:=particle+{0,Cos[particle[[-1]]],Sin[particle[[-1]]],0}+{0,Random[]/10,Random[]/10,Random[]/10};*)


(* ::Input:: *)
(*updateObservation[particle_,observation_]:=*)
(*If[particle[[2]]!=0,*)
(*(y=Round[particle[[3]]];*)
(*x=Round[particle[[2]]];*)
(*Pobscondstate=If[Sea[y,x],If[observation=="Blue",0.999,0.001],*)
(*If[Land[y,x],If[observation=="Yellow"||observation=="White",0.999,0.001],*)
(*If[Danger[y,x],If[observation=="Red",0.999,0.001],*)
(*0.5]]];*)
(*particle*{Pobscondstate,1,1,1}*)
(*),particle]*)
(**)


(* ::Input:: *)
(*resample[particles_]:=Map[{1.0/NoParticles,#[[2]],#[[3]],#[[4]]}&,RandomChoice[Map[First,particles]->particles,NoParticles]]*)


(* ::Input:: *)
(*rastMap=ColorCombine[{ImageApply[0&,Image[map]],EdgeDetect[Image[map]],ImageApply[0&,Image[map]]}];*)


(* ::Input:: *)
(*ParticleToDensity[particles_]:=( *)
(*arr=ConstantArray[0.0,{144,300}];*)
(*For[l=1,l<NoParticles,l++,*)
(*(tp=particles[[l]];*)
(*If[tp[[2]]!=0,*)
(*Assert[tp[[2]]>=1];*)
(*arr[[Round[tp[[3]]],Round[tp[[2]]]]]+= tp[[1]]]*)
(*)];*)
(*arr)*)


(* ::Input:: *)
(*ParticleToDensity1[particles_]:=( *)
(*temp1=Map[{#[[2]],#[[3]],#[[1]]}&,particles];*)
(*blist=BinLists[temp1,{0,300,5},{0,144,5},{0,1,1}];*)
(*tmap=Chop[Map[Total[#[[1,All,3]]]&,Transpose[blist],{2}] ];*)
(*ImageData[ImageResize[Image[tmap],{300}]]*)
(*)*)


(* ::Input:: *)
(*discretise[particles_]:=Transpose[MapAt[Round,Transpose[particles][[1;;3]],{{2},{3}}]]*)


(* ::Input:: *)
(*ParticleToDensity2[particles_,arrayDimensions_: {144,300}]:=Module[{arr},arr=ConstantArray[0,arrayDimensions];*)
(*(arr[[Round[#[[3]]],Round[#[[2]]]]]+=#[[1]])&/@discretise[particles];*)
(*arr]*)


(* ::Input:: *)
(*DisplayParticleMap[particles_]:=( *)
(*Show[ImageAdd[rastMap,Image[ParticleToDensity2[particles]]//ImageAdjust],ImageSize->Large]*)
(*)*)


(* ::Input:: *)
(*GUI[]:=Column[{*)
(*Dynamic[{colour,iter,action,confidence}],*)
(*Dynamic[DisplayParticleMap[currentParticles]]*)
(*}]*)


(* ::Input:: *)
(*confKernel=ConstantArray[1,{12,21,21}];*)


(* ::Input:: *)
(*Process[]:=( *)
(*colour = EV3ReadColourSensor[];*)
(*currentParticles1=Map[updateObservation[#,colour]&,currentParticles];*)
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
(*currentParticles2 = Select[*)
(*Map[updateMotion[#,action]&,currentParticles1],#[[2]]<=300&&#[[2]]>=1&&#[[3]]<=144&&#[[3]]>=1&];*)
(*iter--;*)
(*If[record,AppendTo[blackbox,{colour,action,currentParticles2}]];*)
(*currentParticles3 = resample[currentParticles2];*)
(*currentParticles=currentParticles3;*)
(*(*confMap=ListConvolve[confKernel[[1]],ParticleToDensity[currentParticles],{6,6},0];*)*)
(*confidence=Max[confMap];*)
(*)*)


(* ::Input:: *)
(*Main[iteration_:100]:=*)
(*(currentParticles=initialParticles;*)
(*blackbox={};*)
(*iter=iteration;*)
(*EV3ReadColourSensor[];*)
(*While[iter>0,Process[]*)
(*])*)
(**)


(* ::Input:: *)
(*currentParticles=initialParticles;*)


(* ::Input:: *)
(*Dummy[]:=( *)
(*colSens=0;*)
(*EV3ReadColourSensor[]:=(colSens++;If[8<=colSens<=16,"Yellow","Blue"]);*)
(*EV3StepMotor[_]:=(0)*)
(*)*)

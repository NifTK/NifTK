#! /usr/bin/tclsh 
package require vtk
vtkSphereSource Tip
Tip SetRadius 3.0
Tip SetCenter 0.0 0.0 0.0
Tip SetThetaResolution 20
Tip SetPhiResolution 20

vtkSphereSource IRED_1
IRED_1 SetRadius 3.0
IRED_1 SetThetaResolution 8
IRED_1 SetPhiResolution 8
vtkSphereSource IRED_2
IRED_2 SetRadius 3.0
IRED_2 SetThetaResolution 8
IRED_2 SetPhiResolution 8
vtkSphereSource IRED_3
IRED_3 SetRadius 3.0
IRED_3 SetThetaResolution 8
IRED_3 SetPhiResolution 8
vtkSphereSource IRED_4
IRED_4 SetRadius 3.0
IRED_4 SetThetaResolution 8
IRED_4 SetPhiResolution 8
vtkSphereSource IRED_5
IRED_5 SetRadius 3.0
IRED_5 SetThetaResolution 8
IRED_5 SetPhiResolution 8
vtkSphereSource IRED_6
IRED_6 SetRadius 3.0
IRED_6 SetThetaResolution 8
IRED_6 SetPhiResolution 8

vtkLineSource Join1
vtkLineSource Join2
vtkLineSource Join3
vtkLineSource Join4
vtkLineSource Axis 

IRED_1 SetCenter  8.5315    87.4587   11.2776
IRED_2 SetCenter 31.0887    82.1538    -10.0324
IRED_3 SetCenter  42.0908   125.1522     -9.0685
IRED_4 SetCenter  52.6999   168.4092     -8.0427
IRED_5 SetCenter  30.0982   173.5410   13.1075
IRED_6 SetCenter  19.3680   130.4133   12.2067

Join1 SetPoint1  8.5315    87.4587   11.2776
Join1 SetPoint2 31.0887    82.1538    -10.0324

Join2 SetPoint1 52.6999   168.4092     -8.0427  
Join2 SetPoint2  30.0982   173.5410   13.1075

Join3 SetPoint1  8.5315    87.4587   11.2776 
Join3 SetPoint2 30.0982   173.5410   13.1075

Join4 SetPoint1  31.0887    82.1538    -10.0324
Join4 SetPoint2  52.6999   168.4092    -8.0427

Axis SetPoint1  0 0 0  
Axis SetPoint2  41.3991   170.9751    2.5324

vtkAppendPolyData Appenderer
Appenderer AddInput [ Tip GetOutput ] 
Appenderer AddInput [ IRED_1 GetOutput ] 
Appenderer AddInput [ IRED_2 GetOutput ] 
Appenderer AddInput [ IRED_3 GetOutput ] 
Appenderer AddInput [ IRED_4 GetOutput ] 
Appenderer AddInput [ IRED_5 GetOutput ] 
Appenderer AddInput [ IRED_6 GetOutput ] 
Appenderer AddInput [ Join1 GetOutput ] 
Appenderer AddInput [ Join2 GetOutput ] 
Appenderer AddInput [ Join3 GetOutput ] 
Appenderer AddInput [ Join4 GetOutput ] 
Appenderer AddInput [ Axis GetOutput ] 


vtkXMLPolyDataWriter writer
   writer SetInput [ Appenderer GetOutput ]
   writer SetFileName "Pointer.vtp"
   writer Update
exit



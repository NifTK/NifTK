#! /usr/bin/tclsh 
package require vtk
vtkConeSource Tip
Tip SetRadius 3.0
Tip SetHeight 30.0
Tip SetDirection 0 0 -1
Tip SetCenter 0.0 0.0 15.0

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
vtkSphereSource IRED_7
IRED_7 SetRadius 3.0
IRED_7 SetThetaResolution 8
IRED_7 SetPhiResolution 8
vtkSphereSource IRED_8
IRED_8 SetRadius 3.0
IRED_8 SetThetaResolution 8
IRED_8 SetPhiResolution 8
vtkSphereSource IRED_9
IRED_9 SetRadius 3.0
IRED_9 SetThetaResolution 8
IRED_9 SetPhiResolution 8
vtkSphereSource IRED_10
IRED_10 SetRadius 3.0
IRED_10 SetThetaResolution 8
IRED_10 SetPhiResolution 8
vtkSphereSource IRED_11
IRED_11 SetRadius 3.0
IRED_11 SetThetaResolution 8
IRED_11 SetPhiResolution 8
vtkSphereSource IRED_12
IRED_12 SetRadius 3.0
IRED_12 SetThetaResolution 8
IRED_12 SetPhiResolution 8
vtkSphereSource IRED_13
IRED_13 SetRadius 3.0
IRED_13 SetThetaResolution 8
IRED_13 SetPhiResolution 8
vtkSphereSource IRED_14
IRED_14 SetRadius 3.0
IRED_14 SetThetaResolution 8
IRED_14 SetPhiResolution 8


IRED_1 SetCenter -18.9075 315.0084 -455.7179
IRED_2 SetCenter -11.6686 346.6041 -462.6691
IRED_3 SetCenter -24.0502 335.8520 -459.8770
IRED_4 SetCenter -12.4090 327.7958 -458.4255
IRED_5 SetCenter -117.4456 340.9076 -458.5381
IRED_6 SetCenter -100.9828 281.8653 -447.3397
IRED_7 SetCenter -110.1806 274.3549 -445.3865
IRED_8 SetCenter -100.4033 290.4776 -449.1589
IRED_9 SetCenter -17.4428 294.1782 -450.9983
IRED_10 SetCenter -110.3429 293.9211 -449.4930
IRED_11 SetCenter -114.5420 326.9209 -456.0101
IRED_12 SetCenter -109.1648 337.7424 -458.5861
IRED_13 SetCenter -4.4523 286.7106 -450.1598
IRED_14 SetCenter -9.0563 305.4344 -454.0490


vtkLineSource Axis 
Axis SetPoint1  0 0 0  
Axis SetPoint2  -50  200    -450

vtkAppendPolyData Appenderer
Appenderer AddInput [ Tip GetOutput ] 
Appenderer AddInput [ IRED_1 GetOutput ] 
Appenderer AddInput [ IRED_2 GetOutput ] 
Appenderer AddInput [ IRED_3 GetOutput ] 
Appenderer AddInput [ IRED_4 GetOutput ] 
Appenderer AddInput [ IRED_5 GetOutput ] 
Appenderer AddInput [ IRED_6 GetOutput ] 
Appenderer AddInput [ IRED_7 GetOutput ] 
Appenderer AddInput [ IRED_8 GetOutput ] 
Appenderer AddInput [ IRED_9 GetOutput ] 
Appenderer AddInput [ IRED_10 GetOutput ] 
Appenderer AddInput [ IRED_11 GetOutput ] 
Appenderer AddInput [ IRED_12 GetOutput ] 
Appenderer AddInput [ IRED_13 GetOutput ] 
Appenderer AddInput [ IRED_14 GetOutput ] 
Appenderer AddInput [ Axis GetOutput ] 


vtkXMLPolyDataWriter writer
   writer SetInput [ Appenderer GetOutput ]
   writer SetFileName "Laparosope2.vtp"
   writer Update
exit



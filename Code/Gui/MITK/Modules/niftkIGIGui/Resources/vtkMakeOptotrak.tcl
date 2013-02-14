#! /usr/bin/tclsh 
package require vtk
vtkCylinderSource TopBar1
TopBar1 SetRadius 50
TopBar1 SetHeight 500
TopBar1 CappingOn
TopBar1 SetResolution 20
vtkTransform TopBar1Transform
TopBar1Transform RotateX 0
TopBar1Transform Translate 0 300 0 
vtkTransformPolyDataFilter TopBar1Transformer
TopBar1Transformer SetInput [TopBar1 GetOutput ]
TopBar1Transformer SetTransform TopBar1Transform

vtkCylinderSource TopBar2
TopBar2 SetRadius 50
TopBar2 SetHeight 500
TopBar2 CappingOn
TopBar2 SetResolution 20
vtkTransform TopBar2Transform
TopBar2Transform RotateX 0
TopBar2Transform Translate 0 -300 0 
vtkTransformPolyDataFilter TopBar2Transformer
TopBar2Transformer SetInput [TopBar2 GetOutput ]
TopBar2Transformer SetTransform TopBar2Transform


vtkCylinderSource Neck
Neck SetRadius 50
Neck SetHeight 200
Neck CappingOn
Neck SetResolution 20
vtkTransform NeckTransform
NeckTransform RotateZ 90
NeckTransform Translate 0 150 0 
vtkTransformPolyDataFilter NeckTransformer
NeckTransformer SetInput [Neck GetOutput ]
NeckTransformer SetTransform NeckTransform

vtkCylinderSource Eye
Eye SetRadius 60
Eye SetHeight 100
Eye CappingOff
Eye SetResolution 20
vtkTransform EyeTransform
EyeTransform RotateX 90
EyeTransform Translate 0 0 0 
vtkTransformPolyDataFilter EyeTransformer
EyeTransformer SetInput [Eye GetOutput ]
EyeTransformer SetTransform EyeTransform




vtkSphereSource MiddleEye
MiddleEye SetRadius 60
MiddleEye SetThetaResolution 8
MiddleEye SetPhiResolution 8

vtkSphereSource LeftEye
LeftEye SetRadius 60 
LeftEye SetThetaResolution 8
LeftEye SetPhiResolution 8
vtkSphereSource RightEye
RightEye SetRadius 60
RightEye SetThetaResolution 8
RightEye SetPhiResolution 8

MiddleEye SetCenter 0 0 -20 
LeftEye SetCenter 0 -500 -20
RightEye SetCenter 0 500 -20



vtkAppendPolyData Appenderer
Appenderer AddInput [ TopBar1Transformer GetOutput ] 
Appenderer AddInput [ TopBar2Transformer GetOutput ] 
Appenderer AddInput [ NeckTransformer GetOutput ] 
Appenderer AddInput [ EyeTransformer GetOutput ] 
#Appenderer AddInput [ MiddleEye GetOutput ] 
Appenderer AddInput [ LeftEye GetOutput ] 
Appenderer AddInput [ RightEye GetOutput ] 


vtkXMLPolyDataWriter writer
   writer SetInput [ Appenderer GetOutput ]
   writer SetFileName "Optotrak.vtp"
   writer Update
exit



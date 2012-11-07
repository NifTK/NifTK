#! /usr/bin/tclsh 
package require vtk
# create the calibration matrix
# TODO, make a better interface for this
vtkMatrix4x4 matrix
matrix SetElement 0 0 1.0
matrix SetElement 0 1 0.0 
matrix SetElement 0 2 0.0
matrix SetElement 0 3 0.0

matrix SetElement 1 0 0.0
matrix SetElement 1 1 1.0
matrix SetElement 1 2 0.0
matrix SetElement 1 3 100 

matrix SetElement 2 0 0.0
matrix SetElement 2 1 0.0
matrix SetElement 2 2 1.0
matrix SetElement 2 3 0.0

matrix SetElement 3 0 0.0
matrix SetElement 3 1 0.0
matrix SetElement 3 2 0.0
matrix SetElement 3 3 1.0


vtkTransform Calibtrans
   Calibtrans SetMatrix matrix 
#the narrow part of the body
vtkCylinderSource Body
Body SetRadius 8
Body SetHeight 100
Body SetCenter 0.0 0.0 0.0
Body SetResolution 40
Body CappingOn

vtkTransform BodyTransform 
BodyTransform RotateX 0
BodyTransform Translate 51.696 -60 0
vtkTransformPolyDataFilter BodyTransformer 
BodyTransformer SetInput [Body GetOutput ]
BodyTransformer SetTransform BodyTransform

vtkCylinderSource Cowl
Cowl SetRadius 16.170
Cowl SetHeight 20
Cowl SetCenter 0.0 0.0 0.0
Cowl SetResolution 40
Cowl CappingOn

vtkTransform CowlTransform 
CowlTransform RotateX 0
CowlTransform Translate 45.696 -8.085 0
vtkTransformPolyDataFilter CowlTransformer 
CowlTransformer SetInput [Cowl GetOutput ]
CowlTransformer SetTransform CowlTransform


#A sphere source to represent the transducer,
#centre is 256 * 0.357 / 2 = 45.696, 0 , 0
#radius is r = (4 * 25.4) / (2*pi)
vtkSphereSource Transducer
Transducer SetRadius 16.170 
Transducer SetCenter 45.696 0.0 0.0
Transducer SetThetaResolution 40
Transducer SetPhiResolution 40

#some projection lines
vtkLineSource Projection1
Projection1 SetPoint1 45.696 0.0 0.0
Projection1 SetPoint2 0.0 45.696 0.0

#some projection lines
vtkLineSource Projection2
Projection2 SetPoint1 45.696 0.0 0.0
Projection2 SetPoint2 91.392 45.696 0.0

vtkArcSource Projection3
Projection3 SetPoint1 0.0 45.696 0.0
Projection3 SetPoint2 91.392 45.696 0.0
Projection3 SetCenter 45.696 0.0 0.0
Projection3 SetResolution 40

vtkAppendPolyData Appenderer
Appenderer AddInput [ Projection1 GetOutput ] 
Appenderer AddInput [ Projection2 GetOutput ] 
Appenderer AddInput [ Projection3 GetOutput ] 
Appenderer AddInput [ BodyTransformer GetOutput ] 
Appenderer AddInput [ CowlTransformer GetOutput ] 
Appenderer AddInput [ Transducer GetOutput ] 

vtkTransformPolyDataFilter AppendTransformer 
AppendTransformer SetInput [Appenderer GetOutput ]
AppendTransformer SetTransform Calibtrans



vtkXMLPolyDataWriter writer
   writer SetInput [ AppendTransformer GetOutput ]
   writer SetFileName "4DEC9-5.vtp"
   writer Update
exit



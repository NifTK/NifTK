#! /usr/bin/tclsh 
package require vtk
# create the calibration matrix
set fp [ open "Laparoscope.calib.matrix" "r" ]
set file_data [ read $fp ] 
close $fp
scan $file_data "%f %f %f %f\n%f %f %f %f\n%f %f %f %d\n%f %f %f %f" a00 a01 a02 a03 a10 a11 a12 a13 a20 a21 a22 a23 a30 a31 a32 a33
vtkMatrix4x4 matrix
matrix SetElement 0 0 $a00 
matrix SetElement 0 1 $a01
matrix SetElement 0 2 $a02
matrix SetElement 0 3 $a03

matrix SetElement 1 0 $a10
matrix SetElement 1 1 $a11
matrix SetElement 1 2 $a12
matrix SetElement 1 3 $a13

matrix SetElement 2 0 $a20
matrix SetElement 2 1 $a21
matrix SetElement 2 2 $a22
matrix SetElement 2 3 $a23

matrix SetElement 3 0 $a30
matrix SetElement 3 1 $a31
matrix SetElement 3 2 $a32
matrix SetElement 3 3 $a33


vtkTransform Calibtrans
   Calibtrans SetMatrix matrix 
#the narrow part of the body
vtkCylinderSource Body
Body SetRadius 5
Body SetHeight 400
Body SetCenter 0.0 0.0 0.0
Body SetResolution 40
Body CappingOn

vtkTransform BodyTransform 
BodyTransform RotateX 0
BodyTransform Translate 0 -200 0
vtkTransformPolyDataFilter BodyTransformer 
BodyTransformer SetInput [Body GetOutput ]
BodyTransformer SetTransform BodyTransform

vtkCylinderSource Cowl
Cowl SetRadius 20
Cowl SetHeight 80
Cowl SetCenter 0.0 0.0 0.0
Cowl SetResolution 6 
Cowl CappingOn

vtkTransform CowlTransform 
CowlTransform RotateX 0
CowlTransform Translate 0.0 -40 0
vtkTransformPolyDataFilter CowlTransformer 
CowlTransformer SetInput [Cowl GetOutput ]
CowlTransformer SetTransform CowlTransform


vtkAppendPolyData Appenderer
Appenderer AddInput [ BodyTransformer GetOutput ] 
Appenderer AddInput [ CowlTransformer GetOutput ] 

vtkTransformPolyDataFilter AppendTransformer 
AppendTransformer SetInput [Appenderer GetOutput ]
AppendTransformer SetTransform Calibtrans



vtkXMLPolyDataWriter writer
   writer SetInput [ AppendTransformer GetOutput ]
   writer SetFileName "Laparoscope.vtp"
   writer Update
exit



#! /usr/bin/tclsh 
package require vtk

vtkLineSource ZAxis
vtkLineSource XAxisLHC
vtkLineSource YAxisLHC
vtkLineSource XAxisRHC
vtkLineSource YAxisRHC


ZAxis SetPoint1 0 0 -2000
ZAxis SetPoint2 0 0 2000

XAxisLHC SetPoint1 0 0 -10
XAxisLHC SetPoint2 20 0 -10
XAxisRHC SetPoint1 0 0 10
XAxisRHC SetPoint2 20 0 10

YAxisLHC SetPoint1 0 0 -10
YAxisLHC SetPoint2 0 20 -10
YAxisRHC SetPoint1 0 0 10
YAxisRHC SetPoint2 0 20 10


vtkAppendPolyData Appenderer
Appenderer AddInput [ ZAxis GetOutput ] 
Appenderer AddInput [ XAxisLHC GetOutput ] 
Appenderer AddInput [ XAxisRHC GetOutput ] 
Appenderer AddInput [ YAxisLHC GetOutput ] 
Appenderer AddInput [ YAxisRHC GetOutput ] 


vtkXMLPolyDataWriter writer
   writer SetInput [ Appenderer GetOutput ]
   writer SetFileName "LapLensAxes.vtp"
   writer Update
exit



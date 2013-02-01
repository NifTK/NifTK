#! /usr/bin/tclsh 
package require vtk

vtkLineSource XAxis
XAxis SetPoint1 0 0 0 
XAxis SetPoint2 4000 0 0 

vtkLineSource YAxis
YAxis SetPoint1 0 0 0 
YAxis SetPoint2 0 4000 0 

vtkLineSource ZAxis
ZAxis SetPoint1 0 0 0 
ZAxis SetPoint2  0 0 4000 


vtkXMLPolyDataWriter writer
   writer SetInput [ XAxis GetOutput ]
   writer SetFileName "xAxis.vtp"
   writer Update
   writer SetInput [ YAxis GetOutput ]
   writer SetFileName "yAxis.vtp"
   writer Update
   writer SetInput [ ZAxis GetOutput ]
   writer SetFileName "zAxis.vtp"
   writer Update
exit



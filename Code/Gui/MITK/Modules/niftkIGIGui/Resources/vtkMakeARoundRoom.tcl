#! /usr/bin/tclsh 
package require vtk

vtkSphereSource Body
Body SetRadius 4000
Body SetCenter 0.0 0.0 0.0
Body SetThetaResolution 40
Body SetPhiResolution 40


vtkXMLPolyDataWriter writer
   writer SetInput [ Body  GetOutput ]
   writer SetFileName "BigSphere.vtp"
   writer Update
exit



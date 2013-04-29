#! /usr/bin/tclsh 
package require vtk

vtkCubeSource Body
Body SetXLength 2000
Body SetYLength 2000
Body SetZLength 2000
Body SetCenter 0.0 0.0 0.0
#Body SetResolution 40


vtkXMLPolyDataWriter writer
   writer SetInput [ Body  GetOutput ]
   writer SetFileName "Room.vtp"
   writer Update
exit



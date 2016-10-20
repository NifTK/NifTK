/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
/*!
* \file niftkVTKRandomTransform.cxx
* \page niftkVTKRandomTransform
* \section niftkVTKRandomTransformSummary Uses vtkRandomTransform to create a random 6 DOF transform
*
* This program uses niftk::RandomTransform to make a random 6 DOF transform
*/

#include <niftkVTKRandomTransformCLP.h>
#include <niftkVTKFunctions.h>

#include <niftkConversionUtils.h>
#include <vtkMinimalStandardRandomSequence.h>
#include <vtkMatrix4x4.h>
#include <vtkTransform.h>

int main(int argc, char** argv)
{
  // To parse command line args.
  PARSE_ARGS;

  vtkSmartPointer<vtkMatrix4x4> toCentreMat = vtkSmartPointer<vtkMatrix4x4>::New();
  vtkSmartPointer<vtkTransform> toCentre = vtkSmartPointer<vtkTransform>::New();
  toCentre->SetMatrix(toCentreMat);

  vtkSmartPointer<vtkTransform> StartTrans = vtkSmartPointer<vtkTransform>::New();
    
  vtkSmartPointer<vtkMinimalStandardRandomSequence> Uni_Rand = vtkSmartPointer<vtkMinimalStandardRandomSequence>::New();
  Uni_Rand->SetSeed(time(NULL));
  
  StartTrans = niftk::RandomTransform ( 10.0 , 10.0 , 10.0, 10.0 , 10.0, 10.0, *Uni_Rand, scaleSD);
}

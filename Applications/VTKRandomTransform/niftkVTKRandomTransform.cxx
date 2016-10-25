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
#include <vtkBoxMuellerRandomSequence.h>
#include <vtkMatrix4x4.h>
#include <vtkTransform.h>

int main(int argc, char** argv)
{
  // To parse command line args.
  PARSE_ARGS;

  bool silent = false;

  vtkSmartPointer<vtkMatrix4x4> toCentreMat = vtkSmartPointer<vtkMatrix4x4>::New();
  vtkSmartPointer<vtkTransform> toCentre = vtkSmartPointer<vtkTransform>::New();
  toCentreMat->Identity();
  if ( modelToCentreTransform.length() != 0 )
  {
    toCentreMat = niftk::LoadMatrix4x4FromFile ( modelToCentreTransform, silent );
  }

  toCentre->SetMatrix(toCentreMat);

  vtkSmartPointer<vtkTransform> randomTransform = vtkSmartPointer<vtkTransform>::New();
  vtkSmartPointer<vtkMatrix4x4> randomMatrix = vtkSmartPointer<vtkMatrix4x4>::New();

  vtkSmartPointer<vtkMinimalStandardRandomSequence> Uni_Rand = vtkSmartPointer<vtkMinimalStandardRandomSequence>::New();
  vtkSmartPointer<vtkBoxMuellerRandomSequence> normal_Rand = vtkSmartPointer<vtkBoxMuellerRandomSequence>::New();

  Uni_Rand->SetSeed(seed);

  unsigned int widthOfNumber = static_cast<unsigned int>(std::floor(std::log10(repeats)) + 1);
  for ( unsigned int i = 0 ; i < repeats ; ++i )
  {

    randomTransform = niftk::RandomTransformAboutRemoteCentre ( 10.0 , 10.0 , 10.0, 10.0 , 10.0, 10.0, *normal_Rand,
        toCentre, scaleSD);

    randomMatrix = randomTransform->GetMatrix();

    std::ostringstream outputNumber;
    outputNumber << std::setw(widthOfNumber) << std::setfill ('0') << i;
    std::string output = outputPrefix + outputNumber.str() + ".4x4";

    niftk::SaveMatrix4x4ToFile ( output,*randomMatrix, silent);
  }
}

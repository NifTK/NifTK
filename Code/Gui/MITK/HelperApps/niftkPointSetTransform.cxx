/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <boost/math/special_functions/fpclassify.hpp>
#include <mitkPointSetReader.h>
#include <mitkPointSetWriter.h>
#include <vtkTransform.h>
#include <vtkSmartPointer.h>
#include <vtkMinimalStandardRandomSequence.h>

#include <niftkVTKFunctions.h>
#include <mitkPointUtils.h>
#include <niftkPointSetTransformCLP.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;

  //Read Moving Points
  mitk::PointSetReader::Pointer  movingReader = mitk::PointSetReader::New();
  movingReader->SetFileName(source);
  mitk::PointSet::Pointer movingPoints = mitk::PointSet::New();
  mitk::PointSet::Pointer movedPoints = mitk::PointSet::New();
  movingReader->Update();
  movingPoints = movingReader->GetOutput();
 
  vtkSmartPointer<vtkMatrix4x4> randomMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  vtkMatrix4x4 * userTransform = vtkMatrix4x4::New();
  vtkMatrix4x4 * combinedTransform = vtkMatrix4x4::New();
  userTransform->Identity();
  randomMatrix->Identity();
  if ( (perturbTrans > 0.0) || (perturbRot > 0.0) )
  {
    vtkSmartPointer<vtkTransform> randomTrans = vtkSmartPointer<vtkTransform>::New();
    niftk::RandomTransform ( randomTrans , perturbTrans, perturbTrans ,perturbTrans, 
        perturbRot, perturbRot, perturbRot);
    randomMatrix = randomTrans->GetMatrix();
  }

  if ( transform.length() != 0 ) 
  {
    userTransform = niftk::LoadMatrix4x4FromFile(transform);
  }

  vtkMatrix4x4::Multiply4x4 (userTransform, randomMatrix,combinedTransform);
  mitk::TransformPointsByVtkMatrix ( *movingPoints, *combinedTransform, *movedPoints);
  if ( output.length () != 0 ) 
  {
    mitk::PointSetWriter::Pointer writer = mitk::PointSetWriter::New();
    writer->SetFileName(output);
    writer->SetInput(movedPoints);
    writer->Update();
  }
  return EXIT_SUCCESS;
} 

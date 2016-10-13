/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <mitkPointSetReader.h>
#include <mitkPointSetWriter.h>
#include <vtkTransform.h>
#include <vtkSmartPointer.h>
#include <vtkMinimalStandardRandomSequence.h>

#include <niftkPointUtils.h>
#include <niftkPointSetTransformCLP.h>
#include <niftkVTKFunctions.h>

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
  double scaleSD = -1.0;
  if ( (perturbTrans > 0.0) || (perturbRot > 0.0) )
  {
    vtkSmartPointer<vtkTransform> randomTrans = vtkSmartPointer<vtkTransform>::New();
    vtkSmartPointer<vtkMinimalStandardRandomSequence> uni_Rand = vtkSmartPointer<vtkMinimalStandardRandomSequence>::New();
    uni_Rand->SetSeed(time(NULL));

    randomTrans = niftk::RandomTransform ( perturbTrans, perturbTrans ,perturbTrans,
        perturbRot, perturbRot, perturbRot, uni_Rand, scaleSD );
    randomMatrix = randomTrans->GetMatrix();
  }

  if ( transform.length() != 0 ) 
  {
    userTransform = niftk::LoadMatrix4x4FromFile(transform);
  }

  vtkMatrix4x4::Multiply4x4 (userTransform, randomMatrix,combinedTransform);
  niftk::TransformPointsByVtkMatrix ( *movingPoints, *combinedTransform, *movedPoints);
  if ( output.length () != 0 ) 
  {
    mitk::PointSetWriter::Pointer writer = mitk::PointSetWriter::New();
    writer->SetFileName(output);
    writer->SetInput(movedPoints);
    writer->Update();
  }
  return EXIT_SUCCESS;
} 

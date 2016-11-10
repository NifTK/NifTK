/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <mitkIOUtil.h>
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
  mitk::PointSet::Pointer movingPoints = mitk::IOUtil::LoadPointSet(source);
  mitk::IOUtil::Save(movingPoints, "/dev/shm/movingPointsInOut.mps");
  mitk::PointSet::Pointer movedPoints = mitk::PointSet::New();
 
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
        perturbRot, perturbRot, perturbRot, *uni_Rand, scaleSD );
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
    mitk::IOUtil::Save(movedPoints, output);
  }
  return EXIT_SUCCESS;
} 

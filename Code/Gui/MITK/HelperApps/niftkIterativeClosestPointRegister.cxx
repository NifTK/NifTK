/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <vtkTransform.h>
#include <vtkSmartPointer.h>
#include <vtkMinimalStandardRandomSequence.h>
#include <mitkTestingMacros.h>
#include <mitkPointSetReader.h>
#include <mitkSTLFileReader.h>
#include <mitkDataStorageUtils.h>

#include <niftkVTKFunctions.h>
#include <mitkSurfaceBasedRegistration.h>
#include <niftkIterativeClosestPointRegisterCLP.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
 

  mitk::SurfaceBasedRegistration::Pointer registerer = mitk::SurfaceBasedRegistration::New();
  mitk::DataNode::Pointer fixednode = mitk::DataNode::New();
  mitk::DataNode::Pointer movingnode = mitk::DataNode::New();
  //Read Fixed Points
  mitk::PointSetReader::Pointer  PointReader = mitk::PointSetReader::New();
  PointReader->SetFileName(target);
  mitk::PointSet::Pointer FixedPoints = mitk::PointSet::New();
  mitk::Surface::Pointer FixedSurface = mitk::Surface::New();
  PointReader->Update();
  FixedPoints = PointReader->GetOutput();

  int numberOfPoints = FixedPoints->GetSize();
  if ( numberOfPoints == 0  )
  {
    mitk::STLFileReader::Pointer  FixedSurfaceReader = mitk::STLFileReader::New();
    FixedSurfaceReader->SetFileName(target);
    FixedSurfaceReader->Update();
    FixedSurface = FixedSurfaceReader->GetOutput();
    fixednode->SetData(FixedSurface);
  }
  else
  {
    fixednode->SetData(FixedPoints);
  }

  //Read Moving Surface
  mitk::STLFileReader::Pointer  SurfaceReader = mitk::STLFileReader::New();
  SurfaceReader->SetFileName(source);
  mitk::Surface::Pointer MovingSurface = mitk::Surface::New();
  SurfaceReader->Update();
  MovingSurface = SurfaceReader->GetOutput();
 
  movingnode->SetData(MovingSurface);

  vtkSmartPointer<vtkMatrix4x4> randomMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  if ( (perturbTrans > 0.0) || (perturbRot > 0.0) )
  {
    vtkSmartPointer<vtkTransform> randomTrans = vtkSmartPointer<vtkTransform>::New();
    niftk::RandomTransform ( randomTrans , perturbTrans, perturbTrans ,perturbTrans, 
        perturbRot, perturbRot, perturbRot);
    randomMatrix = randomTrans->GetMatrix();
    mitk::ComposeTransformWithNode(*randomMatrix, movingnode);
  }

  vtkMatrix4x4 * initialTransform = vtkMatrix4x4::New();
  if ( initTrans.length() != 0 ) 
  {
    initialTransform = niftk::LoadMatrix4x4FromFile(initTrans);
    mitk::ComposeTransformWithNode(*initialTransform, movingnode);
  }
   
  vtkMatrix4x4 * resultMatrix = vtkMatrix4x4::New();
  registerer->SetMaximumIterations(maxIterations);
  registerer->SetMaximumNumberOfLandmarkPointsToUse(maxLandmarks);
  
  MITK_INFO << "Starting registration";
  registerer->Update(fixednode, movingnode, *resultMatrix);
  MITK_INFO << "Init" << *initialTransform;
  if ( (perturbTrans > 0.0) || (perturbRot > 0.0) )
  {
    MITK_INFO << "Random" << *randomMatrix;
  }
  MITK_INFO << "Result" << *resultMatrix;
  vtkMatrix4x4 * compound = vtkMatrix4x4::New();
  resultMatrix->Multiply4x4(resultMatrix, initialTransform , compound);
  if ( (perturbTrans > 0.0) || (perturbRot > 0.0) )
  {
    compound->Multiply4x4(compound, randomMatrix , compound);
  }
 
  MITK_INFO << "Full Result " << *compound;
  if ( output.length () != 0 ) 
  {
    niftk::SaveMatrix4x4ToFile(output, *compound);
  }
  return EXIT_SUCCESS;
} 

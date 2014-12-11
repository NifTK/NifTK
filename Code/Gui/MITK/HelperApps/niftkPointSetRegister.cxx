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
#include <mitkTestingMacros.h>
#include <mitkPointBasedRegistration.h>
#include <mitkPointSetReader.h>
#include <mitkDataStorageUtils.h>

#include <niftkVTKFunctions.h>
#include <vtkTransform.h>
#include <vtkSmartPointer.h>
#include <vtkMinimalStandardRandomSequence.h>
#include <niftkPointSetRegisterCLP.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
 

  mitk::PointBasedRegistration::Pointer registerer = mitk::PointBasedRegistration::New();
  mitk::DataNode::Pointer fixednode = mitk::DataNode::New();
  mitk::DataNode::Pointer movingnode = mitk::DataNode::New();
  //Read Fixed Points
  mitk::PointSetReader::Pointer  fixedReader = mitk::PointSetReader::New();
  fixedReader->SetFileName(target);
  mitk::PointSet::Pointer fixedPoints = mitk::PointSet::New();
  fixedReader->Update();
  fixedPoints = fixedReader->GetOutput();

  fixednode->SetData(fixedPoints);

  //Read Moving Points
  mitk::PointSetReader::Pointer  movingReader = mitk::PointSetReader::New();
  movingReader->SetFileName(source);
  mitk::PointSet::Pointer movingPoints = mitk::PointSet::New();
  movingReader->Update();
  movingPoints = movingReader->GetOutput();
 
  movingnode->SetData(movingPoints);

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
  registerer->SetUsePointIDToMatchPoints( usePointIDToMatchPoints);
  registerer->SetUseSVDBasedMethod( useSVDBasedMethod );
  registerer->SetUseICPInitialisation ( useICPInitialisation); 
  
  MITK_INFO << "Starting registration";
  double fre;
  registerer->Update(fixedPoints, movingPoints, *resultMatrix, fre);
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

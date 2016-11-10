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

#include <mitkIOUtil.h>

#include <niftkDataStorageUtils.h>
#include <niftkPointBasedRegistration.h>
#include <niftkPointSetRegisterCLP.h>
#include <niftkPointRegMaths.h>
#include <niftkVTKFunctions.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
 

  niftk::PointBasedRegistration::Pointer registerer = niftk::PointBasedRegistration::New();
  mitk::DataNode::Pointer fixednode = mitk::DataNode::New();
  mitk::DataNode::Pointer movingnode = mitk::DataNode::New();
  //Read Fixed Points
  mitk::PointSet::Pointer fixedPoints = mitk::IOUtil::LoadPointSet ( target  );
  //Read Moving Points
  mitk::PointSet::Pointer movingPoints = mitk::IOUtil::LoadPointSet ( source  );

  fixednode->SetData(fixedPoints);
 
  movingnode->SetData(movingPoints);

  vtkSmartPointer<vtkMatrix4x4> randomMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  double scaleSD = -1.0;
  randomMatrix->Identity();
  if ( (perturbTrans > 0.0) || (perturbRot > 0.0) )
  {
    vtkSmartPointer<vtkTransform> randomTrans = vtkSmartPointer<vtkTransform>::New();
    vtkSmartPointer<vtkMinimalStandardRandomSequence> uni_Rand = vtkSmartPointer<vtkMinimalStandardRandomSequence>::New();
    uni_Rand->SetSeed(time(NULL));
    randomTrans = niftk::RandomTransform ( perturbTrans, perturbTrans ,perturbTrans,
        perturbRot, perturbRot, perturbRot, *uni_Rand, scaleSD );
    randomMatrix = randomTrans->GetMatrix();
    niftk::ComposeTransformWithNode(*randomMatrix, movingnode);
  }

  vtkSmartPointer <vtkMatrix4x4> initialTransform = vtkSmartPointer<vtkMatrix4x4>::New();
  initialTransform->Identity();
  if ( initTrans.length() != 0 ) 
  {
    initialTransform = niftk::LoadMatrix4x4FromFile(initTrans);
    niftk::ComposeTransformWithNode(*initialTransform, movingnode);
  }
   
  vtkSmartPointer <vtkMatrix4x4> resultMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  resultMatrix->Identity();
  registerer->SetUsePointIDToMatchPoints( usePointIDToMatchPoints);
  registerer->SetUseICPInitialisation ( useICPInitialisation); 
  
  if ( usePointIDToMatchPoints )
  {
    MITK_INFO << "Starting FRE = " << niftk::CalculateFiducialRegistrationError ( fixedPoints, movingPoints, *resultMatrix );
  }
  MITK_INFO << "Starting registration";
  double fre = registerer->Update(fixedPoints, movingPoints, *resultMatrix);
  MITK_INFO << "Init" << *initialTransform;
  if ( (perturbTrans > 0.0) || (perturbRot > 0.0) )
  {
    MITK_INFO << "Random" << *randomMatrix;
  }
  MITK_INFO << "Result" << *resultMatrix;
  vtkSmartPointer <vtkMatrix4x4>  compound = vtkSmartPointer<vtkMatrix4x4>::New();
  resultMatrix->Multiply4x4(resultMatrix, initialTransform , compound);
  if ( (perturbTrans > 0.0) || (perturbRot > 0.0) )
  {
    compound->Multiply4x4(compound, randomMatrix , compound);
  }
 
  MITK_INFO << "Full Result " << *compound;
  MITK_INFO << "FRE " << fre;

  if ( output.length () != 0 ) 
  {
    niftk::SaveMatrix4x4ToFile(output, *compound);
  }
  return EXIT_SUCCESS;
} 

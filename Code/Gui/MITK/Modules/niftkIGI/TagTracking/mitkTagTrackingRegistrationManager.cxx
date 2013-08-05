/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTagTrackingRegistrationManager.h"
#include <mitkPointBasedRegistration.h>
#include <mitkPointsAndNormalsBasedRegistration.h>
#include <mitkCoordinateAxesData.h>
#include <mitkSurface.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkIntArray.h>
#include <vtkFloatArray.h>
#include <vtkDataArray.h>
#include <limits>
#include <itkEulerAffineTransform.h>

namespace mitk
{

const char* TagTrackingRegistrationManager::POINTSET_NODE_ID = "Tag Locations";
const char* TagTrackingRegistrationManager::TRANSFORM_NODE_ID = "Tag Transform";

//-----------------------------------------------------------------------------
TagTrackingRegistrationManager::TagTrackingRegistrationManager()
{
  m_ReferenceMatrix = vtkMatrix4x4::New();
  m_ReferenceMatrix->Identity();
}


//-----------------------------------------------------------------------------
TagTrackingRegistrationManager::~TagTrackingRegistrationManager()
{
}


//-----------------------------------------------------------------------------
void TagTrackingRegistrationManager::SetReferenceMatrix(vtkMatrix4x4& referenceMatrix)
{
  m_ReferenceMatrix->DeepCopy(&referenceMatrix);
  this->Modified();
}


//-----------------------------------------------------------------------------
bool TagTrackingRegistrationManager::Update(
    mitk::DataStorage::Pointer& dataStorage,
    mitk::PointSet::Pointer& tagPointSet,
    mitk::PointSet::Pointer& tagNormals,
    mitk::DataNode::Pointer& modelNode,
    const std::string& transformNodeToUpdate,
    const bool useNormals,
    vtkMatrix4x4& registrationMatrix,
    double &fiducialRegistrationError) const
{

  bool isSuccessful = false;
  fiducialRegistrationError = std::numeric_limits<double>::max();
  registrationMatrix.Identity();

  if (modelNode.IsNotNull())
  {
    bool modelIsPointSet = true;
    mitk::PointSet::Pointer modelPointSet = dynamic_cast<mitk::PointSet*>(modelNode->GetData());
    mitk::PointSet::Pointer modelNormals = mitk::PointSet::New();

    if (modelPointSet.IsNull())
    {
      modelIsPointSet = false;

      // model may be a vtk surface, which we need for surface normals.
      mitk::Surface::Pointer surface = dynamic_cast<mitk::Surface*>(modelNode->GetData());
      if (surface.IsNotNull())
      {
        // Here we assume that to register with Points+Normals, we need a VTK model,
        // where the scalar value associated with each point == model ID, and the
        // normal value is stored with each point.  We can convert this to 2 mitk::PointSet.
        vtkPolyData *polyData = surface->GetVtkPolyData();
        if (polyData != NULL)
        {
          vtkPoints *points = polyData->GetPoints();
          vtkPointData * pointData = polyData->GetPointData();
          if (pointData != NULL)
          {
            vtkDataArray *vtkScalars = pointData->GetScalars();
            vtkDataArray *vtkNormals = pointData->GetNormals();

            if (vtkScalars == NULL || vtkNormals == NULL)
            {
              MITK_ERROR << "Surfaces must have scalars containing pointID and normals";
              return isSuccessful;
            }
            if (vtkScalars->GetNumberOfTuples() != vtkNormals->GetNumberOfTuples())
            {
              MITK_ERROR << "Surfaces must have same number of scalars and normals";
              return isSuccessful;
            }

            modelPointSet = mitk::PointSet::New();

            for (int i = 0; i < vtkScalars->GetNumberOfTuples(); ++i)
            {
              mitk::Point3D point;
              mitk::Point3D normal;
              int pointID = static_cast<int>(vtkScalars->GetTuple1(i));

              double tmp[3];
              points->GetPoint(i, tmp);

              point[0] = tmp[0];
              point[1] = tmp[1];
              point[2] = tmp[2];

              double *vtkNormal = vtkNormals->GetTuple3(i);

              normal[0] = vtkNormal[0];
              normal[1] = vtkNormal[1];
              normal[2] = vtkNormal[2];

              modelPointSet->InsertPoint(pointID, point);
              modelNormals->InsertPoint(pointID, normal);
            }
          }
        }
      }
    }

    if (modelPointSet.IsNotNull())
    {
      if (!useNormals || modelIsPointSet)
      {
        // do normal point based registration
        mitk::PointBasedRegistration::Pointer pointBasedRegistration = mitk::PointBasedRegistration::New();
        pointBasedRegistration->SetUsePointIDToMatchPoints(true);
        isSuccessful = pointBasedRegistration->Update(tagPointSet, modelPointSet, registrationMatrix, fiducialRegistrationError);
      }
      else
      {
        // do method that uses normals, and hence can cope with a minimum of only 2 points.
        mitk::PointsAndNormalsBasedRegistration::Pointer pointsAndNormalsRegistration = mitk::PointsAndNormalsBasedRegistration::New();
        isSuccessful = pointsAndNormalsRegistration->Update(tagPointSet, modelPointSet, tagNormals, modelNormals, registrationMatrix, fiducialRegistrationError);
      }

      if (isSuccessful)
      {
        // Also need to create and update a node in DataStorage.
        // We create a mitk::CoordinateAxesData, just like the tracker tools do.
        // So, this makes this plugin function just like a tracker.
        // So, the Tracked Image and Tracked Pointer plugin should be usable by selecting this matrix.
        mitk::DataNode::Pointer coordinateAxesNode = dataStorage->GetNamedNode(transformNodeToUpdate);
        if (coordinateAxesNode.IsNull())
        {
          MITK_ERROR << "Can't find mitk::DataNode with name " << transformNodeToUpdate << std::endl;
          return isSuccessful;
        }
        mitk::CoordinateAxesData::Pointer coordinateAxes = dynamic_cast<mitk::CoordinateAxesData*>(coordinateAxesNode->GetData());
        if (coordinateAxes.IsNotNull())
        {
          coordinateAxes->SetVtkMatrix(registrationMatrix);
          coordinateAxesNode->Modified();
        }

        // Output the matrix parameters, useful for measuring stuff.
        typedef itk::EulerAffineTransform<double, 3, 3> EulerTransform ;
        EulerTransform::Pointer euler = EulerTransform::New();
        EulerTransform::FullAffineTransformType::Pointer affine = EulerTransform::FullAffineTransformType::New();
        EulerTransform::FullAffineTransformType::ParametersType params;
        params.SetSize(affine->GetParameters().GetSize());

        vtkSmartPointer<vtkMatrix4x4> inverseOfReference = vtkMatrix4x4::New();
        vtkMatrix4x4::Invert(m_ReferenceMatrix, inverseOfReference);

        vtkSmartPointer<vtkMatrix4x4> offset = vtkMatrix4x4::New();
        vtkMatrix4x4::Multiply4x4(&registrationMatrix, inverseOfReference, offset);

        for (int i = 0; i < 3; i++)
        {
          for (int j = 0; j < 3; j++)
          {
            params[i*3+j] = offset->GetElement(i, j);
          }
          params[9+i] = offset->GetElement(i,3);
        }
        affine->SetIdentity();
        affine->SetParameters(params);
        euler->SetParametersFromTransform(affine);
        std::cerr << "Matt, Transformation parameters are:" << euler->GetParameters() << std::endl;
      }
    } // end if we have model
  } // end if we have node

  return isSuccessful;

} // end Update method

} // end namespace


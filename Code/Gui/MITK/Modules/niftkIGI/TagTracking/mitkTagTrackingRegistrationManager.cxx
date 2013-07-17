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

namespace mitk
{

//-----------------------------------------------------------------------------
TagTrackingRegistrationManager::TagTrackingRegistrationManager()
{
}


//-----------------------------------------------------------------------------
TagTrackingRegistrationManager::~TagTrackingRegistrationManager()
{
}


//-----------------------------------------------------------------------------
void TagTrackingRegistrationManager::Update(
    mitk::DataStorage::Pointer& dataStorage,
    mitk::PointSet::Pointer& tagPointSet,
    mitk::PointSet::Pointer& tagNormals,
    mitk::DataNode::Pointer& modelNode,
    const std::string& transformNodeToUpdate,
    const bool useNormals,
    vtkMatrix4x4& registrationMatrix) const
{

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
              return;
            }
            if (vtkScalars->GetNumberOfTuples() != vtkNormals->GetNumberOfTuples())
            {
              MITK_ERROR << "Surfaces must have same number of scalars and normals";
              return;
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
        pointBasedRegistration->Update(tagPointSet, modelPointSet, registrationMatrix);
      }
      else
      {
        // do method that uses normals, and hence can cope with a minimum of only 2 points.
        mitk::PointsAndNormalsBasedRegistration::Pointer pointsAndNormalsRegistration = mitk::PointsAndNormalsBasedRegistration::New();
        pointsAndNormalsRegistration->Update(tagPointSet, modelPointSet, tagNormals, modelNormals, registrationMatrix);
      }

      // Also need to create and update a node in DataStorage.
      // We create a mitk::CoordinateAxesData, just like the tracker tools do.
      // So, this makes this plugin function just like a tracker.
      // So, the Tracked Image and Tracked Pointer plugin should be usable by selecting this matrix.
      mitk::DataNode::Pointer coordinateAxesNode = dataStorage->GetNamedNode(transformNodeToUpdate);
      if (coordinateAxesNode.IsNull())
      {
        MITK_ERROR << "Can't find mitk::DataNode with name " << transformNodeToUpdate << std::endl;
        return;
      }
      mitk::CoordinateAxesData::Pointer coordinateAxes = dynamic_cast<mitk::CoordinateAxesData*>(coordinateAxesNode->GetData());
      if (coordinateAxes.IsNull())
      {
        coordinateAxes = mitk::CoordinateAxesData::New();

        // We remove and add to trigger the NodeAdded event,
        // which is not emmitted if the node was added with no data.
        dataStorage->Remove(coordinateAxesNode);
        coordinateAxesNode->SetData(coordinateAxes);
        dataStorage->Add(coordinateAxesNode);
      }
      coordinateAxes->SetVtkMatrix(registrationMatrix);

    } // end if we have model
  } // end if we have node
} // end Update method

} // end namespace


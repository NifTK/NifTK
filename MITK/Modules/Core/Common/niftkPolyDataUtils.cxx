/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPolyDataUtils.h"

#include <niftkVTKBackfaceCullingFilter.h>
#include <niftkDataStorageUtils.h>
#include <mitkSurface.h>
#include <mitkExceptionMacro.h>
#include <vtkSmartPointer.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkPolyDataNormals.h>
#include <vtkAppendPolyData.h>
#include <vtkCellArray.h>
#include <vtkCell.h>

namespace niftk
{

//-----------------------------------------------------------------------------
void PointSetToPolyData(const mitk::PointSet::Pointer& pointsIn,
                        vtkPolyData& polyOut)
{
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

  for (mitk::PointSet::PointsConstIterator i = pointsIn->Begin(); i != pointsIn->End(); ++i)
  {
    // we need the point in world-coordinates! i.e. take its index-to-world transformation into account.
    // so instead of i->Value() we go via GetPointIfExists(i->Id(), ...)
    mitk::PointSet::PointType p = i->Value();
    pointsIn->GetPointIfExists(i->Index(), &p);
    points->InsertNextPoint(p[0], p[1], p[2]);
  }

  if (pointsIn->GetSize() != points->GetNumberOfPoints())
  {
    mitkThrow() << "Number of points in mitk::PointSet (" << pointsIn->GetSize()
                << ") != number of points in output (" <<  points->GetNumberOfPoints()
                << ")" << std::endl;
  }

  // What happens if polyOut contains vertices, triangles, poly-lines etc.
  // I think its safer to start again, so I'm calling Initialize().
  polyOut.Initialize();
  polyOut.SetPoints(points);
}


//-----------------------------------------------------------------------------
void NodeToPolyData(const mitk::DataNode::Pointer& node,
                    vtkPolyData& polyOut,
                    const mitk::DataNode::Pointer& cameraNode,
                    bool flipNormals)
{
  if (node.IsNull())
  {
    mitkThrow() << "In ICPBasedRegistration::NodeToPolyData, node is NULL";
  }

  mitk::PointSet::Pointer points = dynamic_cast<mitk::PointSet*>(node->GetData());
  mitk::Surface::Pointer surface = dynamic_cast<mitk::Surface*>(node->GetData());

  if (points.IsNotNull())
  {
    niftk::PointSetToPolyData(points, polyOut);
  }
  else if (surface.IsNotNull())
  {
    // No smartpointer for polyTemp!
    // mitk is handing us back a raw pointer, and uses raw pointers internally.
    vtkPolyData *polyTemp = surface->GetVtkPolyData();

    vtkSmartPointer<vtkMatrix4x4> indexToWorld = vtkSmartPointer<vtkMatrix4x4>::New();
    niftk::GetCurrentTransformFromNode(node, *indexToWorld);

    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->SetMatrix(indexToWorld);

    if (cameraNode.IsNull())
    {
      vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter= vtkSmartPointer<vtkTransformPolyDataFilter>::New();
      transformFilter->SetInputDataObject(polyTemp);
      transformFilter->SetOutput(&polyOut);
      transformFilter->SetTransform(transform);
      transformFilter->Update();
    }
    else
    {
      vtkSmartPointer<vtkPolyData> transformedPoly = vtkSmartPointer<vtkPolyData>::New();

      vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter= vtkSmartPointer<vtkTransformPolyDataFilter>::New();
      transformFilter->SetInputDataObject(polyTemp);
      transformFilter->SetOutput(transformedPoly);
      transformFilter->SetTransform(transform);
      transformFilter->Update();

      vtkSmartPointer<vtkPolyData> normalsPoly = vtkSmartPointer<vtkPolyData>::New();

      vtkSmartPointer<vtkPolyDataNormals> normalFilter = vtkSmartPointer<vtkPolyDataNormals>::New();
      normalFilter->SetInputDataObject(transformedPoly);
      normalFilter->SetOutput(normalsPoly);
      normalFilter->ComputeCellNormalsOn();
      normalFilter->ComputePointNormalsOn();
      normalFilter->AutoOrientNormalsOn();
      normalFilter->ConsistencyOn();
      normalFilter->SetFlipNormals(flipNormals ? 1 : 0);
      normalFilter->Update();

      vtkSmartPointer<vtkMatrix4x4> camtxf = vtkSmartPointer<vtkMatrix4x4>::New();
      niftk::GetCurrentTransformFromNode(cameraNode, *camtxf);

      vtkSmartPointer<niftk::BackfaceCullingFilter> backfacecullingfilter =
          vtkSmartPointer<niftk::BackfaceCullingFilter>::New();
      backfacecullingfilter->SetInputDataObject(normalsPoly);
      backfacecullingfilter->SetOutput(&polyOut);
      backfacecullingfilter->SetCameraPosition(camtxf);

      // this should call Update() instead of Execute().
      // but vtk6 has changed in some way that the filter's Execute() is no longer called.
      backfacecullingfilter->Execute();
    }
  }
  else
  {
    mitkThrow() << "In ICPBasedRegistration::NodeToPolyData, "
                << "node is neither mitk::PointSet or mitk::Surface";
  }
}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkPolyData> MergePolyData(const std::vector<mitk::DataNode::Pointer>& nodes,
                                           const mitk::DataNode::Pointer& cameraNode,
                                           bool flipNormals
                                          )
{
  bool weAreDoingSurface = true;
  for (int i = 0; i < nodes.size(); i++)
  {
    if (dynamic_cast<mitk::PointSet*>(nodes[i]->GetData()) != nullptr)
    {
      weAreDoingSurface = false;
    }
  }

  if (weAreDoingSurface)
  {
    std::vector<vtkSmartPointer<vtkPolyData> > vectorOfPolys;
    vtkSmartPointer<vtkAppendPolyData> filter = vtkSmartPointer<vtkAppendPolyData>::New();

    for (int i = 0; i < nodes.size(); i++)
    {
      vtkSmartPointer<vtkPolyData> poly = vtkSmartPointer<vtkPolyData>::New();
      niftk::NodeToPolyData(nodes[i], *poly, cameraNode, flipNormals);
      vectorOfPolys.push_back(poly);
      filter->AddInputData(vectorOfPolys[i]);
      MITK_INFO << "MergePolyData: added surface with " << poly->GetNumberOfPoints() << " points.";
    }

    filter->Update();

    vtkSmartPointer<vtkPolyData> result = filter->GetOutput();
    return result;
  }
  else
  {
    // If a vector has at least one point set, we convert all of
    // them ourselves, just adding points, not cells.

    vtkSmartPointer<vtkPolyData> outputPoly = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPoints> outputPoints = vtkSmartPointer<vtkPoints>::New();
    for (int i = 0; i < nodes.size(); i++)
    {
      vtkSmartPointer<vtkPolyData> poly = vtkSmartPointer<vtkPolyData>::New();
      niftk::NodeToPolyData(nodes[i], *poly, cameraNode, flipNormals);
      vtkPoints *tmpPoints = poly->GetPoints();
      for (vtkIdType j = 0; j < tmpPoints->GetNumberOfPoints(); j++)
      {
        double* tmpPoint = tmpPoints->GetPoint(j);
        outputPoints->InsertNextPoint(tmpPoint[0], tmpPoint[1], tmpPoint[2]);
      }
      MITK_INFO << "MergePolyData: added point-set with " << tmpPoints->GetNumberOfPoints() << " points.";
    }
    outputPoly->SetPoints(outputPoints);
    return outputPoly;
  }
}

} // end namespace

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkBifurcationToPointSet.h"
#include <mitkExceptionMacro.h>
#include <mitkPointUtils.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkPointLocator.h>
#include <vtkCleanPolyData.h>
#include <vtkCellArray.h>
#include <vtkIdList.h>
#include <vtkMath.h>

namespace mitk
{

//-----------------------------------------------------------------------------
BifurcationToPointSet::BifurcationToPointSet()
{
}


//-----------------------------------------------------------------------------
BifurcationToPointSet::~BifurcationToPointSet()
{
}


//-----------------------------------------------------------------------------
void BifurcationToPointSet::Update(const std::vector<vtkPolyData*> polyDatas,
                                   mitk::PointSet& pointSet
                                  )
{

  if (polyDatas.size() == 0)
  {
    mitkThrow() << "Invalid input, array of vtkPolyData is empty." << std::endl;
  }

  pointSet.Clear();

  for (unsigned int polyDataCounter = 0; polyDataCounter < polyDatas.size(); polyDataCounter++)
  {
    if (polyDatas[polyDataCounter] == NULL)
    {
      continue;
    }

    vtkSmartPointer<vtkCleanPolyData> cleaner = vtkSmartPointer<vtkCleanPolyData>::New();
    cleaner->SetInputData(polyDatas[polyDataCounter]);
    cleaner->PointMergingOn();
    cleaner->Update();

    vtkPolyData* poly = cleaner->GetOutput();
    assert(poly);
    poly->BuildLinks();
    vtkCellArray* lines = poly->GetLines();
    assert(lines);
    vtkPoints *points = poly->GetPoints();
    assert(points);

    mitk::PointSet::Pointer pointSetToAverage = mitk::PointSet::New();

    double point[3];
    mitk::Point3D mitkPoint;

    for (vtkIdType pointCounter = 0; pointCounter < points->GetNumberOfPoints(); pointCounter++)
    {
      points->GetPoint(pointCounter, point);

      vtkSmartPointer<vtkIdList> cellIdsContainingAPoint = vtkSmartPointer<vtkIdList>::New();
      poly->GetPointCells(pointCounter, cellIdsContainingAPoint);

      std::set<vtkIdType> after;

      for (vtkIdType cellCounter = 0; cellCounter < cellIdsContainingAPoint->GetNumberOfIds(); cellCounter++)
      {
        vtkSmartPointer<vtkIdList> allPointsInACell = vtkSmartPointer<vtkIdList>::New();
        poly->GetCellPoints(cellIdsContainingAPoint->GetId(cellCounter), allPointsInACell);

        for (vtkIdType pointIterator = 0; pointIterator < allPointsInACell->GetNumberOfIds(); pointIterator++)
        {
          if (   allPointsInACell->GetId(pointIterator) == pointCounter
              && pointIterator != (allPointsInACell->GetNumberOfIds()-1))
          {
            after.insert(allPointsInACell->GetId(pointIterator + 1));
          }
        }
      }

      if (after.size() > 1)
      {
        std::set<vtkIdType>::iterator iter;
        for (iter = after.begin(); iter != after.end(); iter++)
        {
          points->GetPoint(*iter, point);
          mitkPoint[0] = point[0];
          mitkPoint[1] = point[1];
          mitkPoint[2] = point[2];
          pointSetToAverage->InsertPoint(pointSetToAverage->GetSize(), mitkPoint);
        }
        mitk::Point3D centroid = mitk::ComputeCentroid(*pointSetToAverage);
        pointSet.InsertPoint(pointSet.GetSize(), centroid);
      }
      pointSetToAverage->Clear();

    } // end foreach point
  } // end foreach polyData
} // end Update function

} // end namespace


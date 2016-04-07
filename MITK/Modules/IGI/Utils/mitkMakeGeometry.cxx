/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMakeGeometry.h"
#include <mitkFileIOUtils.h>

#include <vtkCubeSource.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkIdList.h>
#include <vtkCleanPolyData.h>
#include <mitkPointUtils.h>
#include <niftkVTKIGIGeometry.h>

//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakeLaparoscope ( std::string RigidBodyFilename, 
    std::string LeftHandeyeFilename, 
    std::string RightHandeyeFilename, 
    std::string CentreHandeyeFilename,
    bool AddCrossHairs,
    float TrackerMarkerRadius )
{
  niftk::VTKIGIGeometry maker;
  vtkSmartPointer<vtkPolyData> laparoscope = maker.MakeLaparoscope(RigidBodyFilename, LeftHandeyeFilename, RightHandeyeFilename, CentreHandeyeFilename, AddCrossHairs, TrackerMarkerRadius);
  mitk::Surface::Pointer surface = mitk::Surface::New();
  surface->SetVtkPolyData(laparoscope);
  return surface;
}

//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakePointer ( std::string rigidBodyFilename, std::string handeyeFilename ) 
{
  niftk::VTKIGIGeometry maker;
  vtkSmartPointer<vtkPolyData> pointer = maker.MakePointer(rigidBodyFilename, handeyeFilename);
  mitk::Surface::Pointer surface = mitk::Surface::New();
  surface->SetVtkPolyData(pointer);
  return surface;
}


//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakeReference ( std::string rigidBodyFilename, std::string handeyeFilename ) 
{
  niftk::VTKIGIGeometry maker;
  vtkSmartPointer<vtkPolyData> reference = maker.MakeReference(rigidBodyFilename, handeyeFilename);
  mitk::Surface::Pointer surface = mitk::Surface::New();
  surface->SetVtkPolyData(reference);
  return surface;
}


//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakeReferencePolaris ( std::string rigidBodyFilename, std::string handeyeFilename ) 
{
  niftk::VTKIGIGeometry maker;
  vtkSmartPointer<vtkPolyData> reference = maker.MakeReferencePolaris(rigidBodyFilename, handeyeFilename);
  mitk::Surface::Pointer surface = mitk::Surface::New();
  surface->SetVtkPolyData(reference);
  return surface;
}


//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakeAWall ( const int& whichwall, const float& size, 
   const float& xOffset,  const float& yOffset,  const float& zOffset , 
   const float& thickness ) 
{
  niftk::VTKIGIGeometry maker;
  vtkSmartPointer<vtkPolyData> wall = maker.MakeAWall(
      whichwall, size, xOffset, yOffset, zOffset, thickness);

  mitk::Surface::Pointer surface = mitk::Surface::New();
  surface->SetVtkPolyData(wall);
  return surface;
}


//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakeXAxes( const float& length ,const bool& symmetric )
{
  niftk::VTKIGIGeometry maker;
  vtkSmartPointer<vtkPolyData> axis = maker.MakeXAxes(length,symmetric);
  mitk::Surface::Pointer surface = mitk::Surface::New();
  surface->SetVtkPolyData(axis);
  return surface;
}


//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakeYAxes( const float& length ,const bool& symmetric )
{
  niftk::VTKIGIGeometry maker;
  vtkSmartPointer<vtkPolyData> axis = maker.MakeYAxes(length,symmetric);
  mitk::Surface::Pointer surface = mitk::Surface::New();
  surface->SetVtkPolyData(axis);
  return surface;
}


//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakeZAxes( const float& length ,const bool& symmetric )
{
  niftk::VTKIGIGeometry maker;
  vtkSmartPointer<vtkPolyData> axis = maker.MakeZAxes(length,symmetric);
  mitk::Surface::Pointer surface = mitk::Surface::New();
  surface->SetVtkPolyData(axis);
  return surface;
}


//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakeLapLensAxes()
{
  niftk::VTKIGIGeometry maker;
  vtkSmartPointer<vtkPolyData> axes = maker.MakeLapLensAxes();
  mitk::Surface::Pointer surface = mitk::Surface::New();
  surface->SetVtkPolyData(axes);
  return surface;
}

//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakeOptotrak( const float & width )
{
  niftk::VTKIGIGeometry maker;
  vtkSmartPointer<vtkPolyData> optotrak = maker.MakeOptotrak(width);
  mitk::Surface::Pointer surface = mitk::Surface::New();
  surface->SetVtkPolyData(optotrak);
  return surface;
}

//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakePolaris( const float & width )
{
  niftk::VTKIGIGeometry maker;
  vtkSmartPointer<vtkPolyData> polaris = maker.MakeOptotrak(width, true);
  mitk::Surface::Pointer surface = mitk::Surface::New();
  surface->SetVtkPolyData(polaris);
  return surface;
}

//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakeTransrectalUSProbe(std::string handeyeFilename )
{
  niftk::VTKIGIGeometry maker;
  vtkSmartPointer<vtkPolyData> probe = maker.MakeTransrectalUSProbe(handeyeFilename);
  mitk::Surface::Pointer surface = mitk::Surface::New();
  surface->SetVtkPolyData(probe);
  return surface;
}

//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakeMonitor( )
{
  niftk::VTKIGIGeometry maker;
  vtkSmartPointer<vtkPolyData> monitor = maker.MakeMonitor();
  mitk::Surface::Pointer surface = mitk::Surface::New();
  surface->SetVtkPolyData(monitor);
  return surface;
}


//-----------------------------------------------------------------------------
mitk::PointSet::Pointer MakePointSetOfBifurcations(const std::vector<vtkPolyData*> polyDatas)
{
  if (polyDatas.size() == 0)
  {
    mitkThrow() << "Invalid input, array of vtkPolyData is empty." << std::endl;
  }

  mitk::PointSet::Pointer pointSet = mitk::PointSet::New();

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
        pointSet->InsertPoint(pointSet->GetSize(), centroid);
      }
      pointSetToAverage->Clear();

    } // end foreach point
  } // end foreach polyData

  // Return the newly created point set.
  return pointSet;
}

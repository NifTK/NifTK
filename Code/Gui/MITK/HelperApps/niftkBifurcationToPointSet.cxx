/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkBifurcationToPointSetCLP.h>
#include <mitkExceptionMacro.h>
#include <mitkPointSet.h>
#include <mitkIOUtil.h>
#include <mitkPointUtils.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkSmartPointer.h>
#include <vtkPointLocator.h>
#include <vtkCleanPolyData.h>
#include <vtkCellArray.h>
#include <vtkIdList.h>
#include <vtkMath.h>

#include <map>

int main(int argc, char** argv)
{
  PARSE_ARGS;
 
  int returnStatus = EXIT_FAILURE;

  if ( input.length() == 0 || output.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
    reader->SetFileName(input.c_str());
    reader->Update();

    vtkSmartPointer<vtkCleanPolyData> cleaner = vtkSmartPointer<vtkCleanPolyData>::New();
    cleaner->SetInputData(reader->GetOutput());
    cleaner->PointMergingOn();
    cleaner->Update();

    vtkPolyData* poly = cleaner->GetOutput();
    assert(poly);
    poly->BuildLinks();
    vtkCellArray* lines = poly->GetLines();
    assert(lines);
    vtkPoints *points = poly->GetPoints();
    assert(points);

    mitk::PointSet::Pointer finalPointSet = mitk::PointSet::New();
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
        finalPointSet->InsertPoint(finalPointSet->GetSize(), centroid);
      }
      pointSetToAverage->Clear();
    }

    if (!mitk::IOUtil::SavePointSet(finalPointSet, output))
    {
      mitkThrow() << "Failed to save file" << output << std::endl;
    }

    // Done
    returnStatus = EXIT_SUCCESS;
  }
  catch (const mitk::Exception& e)
  {
    std::cerr << "Caught MITK Exception:" << e.GetDescription() << std::endl
                 << "in:" << e.GetFile() << std::endl
                 << "at line:" << e.GetLine() << std::endl;
    returnStatus = -1;
  }
  catch (std::exception& e)
  {
    std::cerr << "Caught std::exception:" << e.what();
    returnStatus = -2;
  }
  catch (...)
  {
    std::cerr << "Caught unknown exception:";
    returnStatus = -3;
  }
  return returnStatus;
} 

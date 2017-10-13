/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkSmartPointer.h>
#include <vtkTubeFilter.h>
#include <vtkTriangleFilter.h>
#include <mitkExceptionMacro.h>
#include <mitkPointSet.h>
#include <mitkIOUtil.h>

#include <mitkMakeGeometry.h>
#include <niftkBifurcationToPointSetCLP.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if ( input.length() == 0 || outputPointSet.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
    reader->SetFileName(input.c_str());
    reader->Update();

    std::vector<vtkPolyData*> polyDatas;
    polyDatas.push_back(reader->GetOutput());

    mitk::PointSet::Pointer finalPointSet = MakePointSetOfBifurcations(polyDatas);

    if (!mitk::IOUtil::SavePointSet(finalPointSet, outputPointSet))
    {
      mitkThrow() << "Failed to save file" << outputPointSet << std::endl;
    }

    if (outputTubes.size() > 0 && polyDatas.size() > 0)
    {
      if (polyDatas.size() > 1)
      {
        mitkThrow() << "Poly Data array size is > 1. I wasn't expecting this." << std::endl;
      }

      vtkSmartPointer<vtkTubeFilter> tubeFilter = vtkSmartPointer<vtkTubeFilter>::New();
      tubeFilter->SetInputData(polyDatas[0]);
      tubeFilter->SetRadius(0.1);
      tubeFilter->Update();

      vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
      triangleFilter->SetInputData(tubeFilter->GetOutput());
      triangleFilter->Update();

      vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
      writer->SetInputData(triangleFilter->GetOutput());
      writer->SetFileName(outputTubes.c_str());
      writer->Update();

    }

    returnStatus = EXIT_SUCCESS;
  }
  catch (mitk::Exception& e)
  {
    MITK_ERROR << "Caught mitk::Exception: " << e.GetDescription() << ", from:" << e.GetFile() << "::" << e.GetLine() << std::endl;
    returnStatus = EXIT_FAILURE + 100;
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception: " << e.what() << std::endl;
    returnStatus = EXIT_FAILURE + 101;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:" << std::endl;
    returnStatus = EXIT_FAILURE + 102;
  }
  return returnStatus;
} 

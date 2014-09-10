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
#include <mitkBifurcationToPointSet.h>
#include <vtkPolyDataReader.h>
#include <vtkSmartPointer.h>

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

    mitk::PointSet::Pointer finalPointSet = mitk::PointSet::New();

    std::vector<vtkPolyData*> polyDatas;
    polyDatas.push_back(reader->GetOutput());

    mitk::BifurcationToPointSet::Pointer converter = mitk::BifurcationToPointSet::New();
    converter->Update(polyDatas, *finalPointSet);

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

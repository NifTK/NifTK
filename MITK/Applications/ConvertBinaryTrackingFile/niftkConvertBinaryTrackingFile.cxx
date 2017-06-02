/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#include <niftkConvertBinaryTrackingFileCLP.h>
#include <niftkIGIDataSourceI.h>
#include <niftkFileHelper.h>
#include <niftkMITKMathsUtils.h>
#include <niftkVTKFunctions.h>
#include <mitkExceptionMacro.h>
#include <vtkSmartPointer.h>
#include <iostream>
#include <sstream>
#include <cstdlib>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  try
  {
    if (inputFile.empty() ||
        (outputDirectory.empty() && outputFile.empty())
       )
    {
      commandLine.getOutput()->usage(commandLine);
      return returnStatus;
    }

    if (!outputDirectory.empty() && !outputFile.empty())
    {
      mitkThrow() << "Specify either output file or directory.";
    }
    if (!outputDirectory.empty() && !niftk::DirectoryExists(outputDirectory))
    {
      mitkThrow() << "Directory:" << outputDirectory << ", doesn't exist!";
    }

    std::ifstream ifs(inputFile, std::ios::binary | std::ios::in);
    if (!ifs.is_open())
    {
      mitkThrow() << "Failed to open input file:" << inputFile;
    }

    std::ofstream ofs;
    if (!outputFile.empty())
    {
      ofs.open(outputFile, std::ios::out);
    }
    if (!ofs.is_open())
    {
      mitkThrow() << "Failed to open output file:" << outputFile;
    }

    unsigned long int counter = 0;
    int modulo = skip + 1;
    vtkSmartPointer<vtkMatrix4x4> mat = vtkSmartPointer<vtkMatrix4x4>::New();
    while(ifs.good())
    {
      niftk::IGIDataSourceI::IGITimeType time;
      std::pair<mitk::Point4D, mitk::Vector3D> transform;
      ifs >> time;
      ifs >> transform.first[0];
      ifs >> transform.first[1];
      ifs >> transform.first[2];
      ifs >> transform.first[3];
      ifs >> transform.second[0];
      ifs >> transform.second[1];
      ifs >> transform.second[2];
      if (ifs.good() && counter % modulo == 0)
      {
        if (!outputDirectory.empty())
        {
          niftk::ConvertRotationAndTranslationToMatrix(transform.first, transform.second, *mat);
          std::ostringstream oss;
          oss << outputDirectory << niftk::GetFileSeparator() << time << ".txt";
          bool successful = niftk::SaveMatrix4x4ToFile(oss.str(), *mat);
          if (!successful)
          {
            mitkThrow() << "Failed to write matrix to:" << oss.str();
          }
        }
        else if (!outputFile.empty())
        {
          ofs << time << " ";
          ofs << transform.first[0] << " ";
          ofs << transform.first[1] << " ";
          ofs << transform.first[2] << " ";
          ofs << transform.first[3] << " ";
          ofs << transform.second[0] << " ";
          ofs << transform.second[1] << " ";
          ofs << transform.second[2] << " ";
          ofs << std::endl;
        }
      }
      else
      {
        mitkThrow() << "Failed to parse entry:" << counter;
      }
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

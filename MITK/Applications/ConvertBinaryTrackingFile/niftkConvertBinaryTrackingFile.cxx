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
      if (!ofs.is_open())
      {
        mitkThrow() << "Failed to open output file:" << outputFile;
      }
    }

    //mitk::StandaloneDataStorage::Pointer dataStorage = mitk::StandaloneDataStorage::New();
    //QString name = "TQRT Converter";
    //niftk::IGISingleFileBackend::Pointer backend = niftk::IGISingleFileBackend::New("TQRT Conerter", dataStorage.GetPointer());
    //backend->CheckFileHeader (ifs);
    // Let's put getfileheader checkfile header into niftkFileHelper. It will give a smaller app

    unsigned long int counter = 0;
    int modulo = skip + 1;
    vtkSmartPointer<vtkMatrix4x4> mat = vtkSmartPointer<vtkMatrix4x4>::New();

    if ( headerSize > 0 )
    {
      niftk::CheckTQRDFileHeader ( ifs, headerSize );
    }
    while(ifs.good())
    {
      niftk::IGIDataSourceI::IGITimeType time;
      std::pair<mitk::Point4D, mitk::Vector3D> transform;

      ifs.read(reinterpret_cast<char*>(&time), sizeof(niftk::IGIDataSourceI::IGITimeType));
      ifs.read(reinterpret_cast<char*>(&transform.first[0]), sizeof(transform.first[0]));
      ifs.read(reinterpret_cast<char*>(&transform.first[1]), sizeof(transform.first[1]));
      ifs.read(reinterpret_cast<char*>(&transform.first[2]), sizeof(transform.first[2]));
      ifs.read(reinterpret_cast<char*>(&transform.first[3]), sizeof(transform.first[3]));
      ifs.read(reinterpret_cast<char*>(&transform.second[0]), sizeof(transform.second[0]));
      ifs.read(reinterpret_cast<char*>(&transform.second[1]), sizeof(transform.second[1]));
      ifs.read(reinterpret_cast<char*>(&transform.second[2]), sizeof(transform.second[2]));

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

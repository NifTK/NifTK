/*===================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center,
Division of Medical and Biological Informatics.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.

See LICENSE.txt or http://www.mitk.org for details.

===================================================================*/

#include "mitkPointSet.h"
#include <mitkTestingMacros.h>
#include <mitkFileReaderRegistry.h>
#include <mitkFileWriterRegistry.h>
#include <mitkFileWriterSelector.h>
#include <mitkCoordinateAxesData.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <usModuleInitialization.h>

/**
 *  @brief Tests for reading/writing .4x4 files via Services.
 */
int niftkCoordinateAxesDataReaderWriterServiceTest(int argc , char* argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("niftkCoordinateAxesDataReaderWriterServiceTest")
  MITK_TEST_CONDITION_REQUIRED(argc == 2,"Testing invocation.")

  // Create matrix to store. The mitk::CoordinateAxesData only store the 3x3 rotation
  // matrix plus the offset. So the bottom row is always as per Identity matrix.
  vtkSmartPointer<vtkMatrix4x4> matrix = vtkSmartPointer<vtkMatrix4x4>::New();
  matrix->Identity();
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      matrix->SetElement(i, j, i+j);
    }
  }
  MITK_TEST_CONDITION_REQUIRED(matrix != NULL,"Testing matrix instantiation.")

  mitk::CoordinateAxesData::Pointer cad = mitk::CoordinateAxesData::New();
  cad->SetVtkMatrix(*matrix);

  // Get CoordinateAxesData writer(s), check for only 1.
  mitk::FileWriterSelector writerSelector(cad.GetPointer());
  std::vector<mitk::FileWriterSelector::Item> writers = writerSelector.Get();
  MITK_TEST_CONDITION_REQUIRED(writers.size() == 1, "Testing for 1 registered writers")

  // Test for exception handling. If invalid (non-writable) file, exception must be thrown.
/*
  try
  {
    mitk::IFileWriter* writer = writers[0].GetWriter();
    writer->SetInput(cad);
    writer->SetOutputLocation("/usr/bin");
    writer->Write();
    MITK_TEST_FAILED_MSG( << "itk::ExceptionObject expected, as /usr/bin should be unwritable" )
  }
  catch (const itk::ExceptionObject&)
  {
    // this is expected
  }
  catch(...)
  {
    // This means that a wrong exception (i.e. no itk:Exception) has been thrown
    MITK_TEST_FAILED_MSG( << "Wrong exception (i.e. no itk:Exception) caught during write [FAILED]")
  }
*/
  try
  {
    mitk::IFileWriter* writer = writers[0].GetWriter();
    writer->SetInput(cad);
    writer->SetOutputLocation(argv[1]);

    std::cout << "Writing to:" << argv[1] << std::endl;
    writer->Write();
  }
  catch (const itk::ExceptionObject&)
  {
    MITK_TEST_FAILED_MSG( << "Writing to file failed, which it should not.")
  }
  catch(...)
  {
    MITK_TEST_FAILED_MSG( << "Wrong exception (i.e. no itk:Exception) caught during write.")
  }

  // Check for 1 reader
  mitk::FileReaderRegistry readerRegistry;
  std::vector<mitk::IFileReader*> readers = readerRegistry.GetReaders(mitk::FileReaderRegistry::GetMimeTypeForFile("4x4"));
  MITK_TEST_CONDITION_REQUIRED(readers.size() == 1, "Testing for 1 registered readers")

  try
  {
    mitk::IFileReader* reader = readers[0];
    reader->SetInput(argv[1]);
    MITK_TEST_CONDITION_REQUIRED( reader->GetConfidenceLevel() == mitk::IFileReader::Supported, "Testing confidence level with valid input file name!");
    std::vector<mitk::BaseData::Pointer> data = reader->Read();
    MITK_TEST_CONDITION_REQUIRED( !data.empty(), "Testing non-empty data with valid input file name!");

    mitk::CoordinateAxesData* cadFromFile = dynamic_cast<mitk::CoordinateAxesData*>(data[0].GetPointer());
    MITK_TEST_CONDITION_REQUIRED(cadFromFile != NULL,"Check if reader returned null")

    vtkSmartPointer<vtkMatrix4x4> matrixFromFile = vtkSmartPointer<vtkMatrix4x4>::New();
    cadFromFile->GetVtkMatrix(*matrixFromFile);

    for (int i = 0; i < 4; ++i)
    {
      for (int j = 0; j < 4; ++j)
      {
        MITK_TEST_CONDITION_REQUIRED(matrix->GetElement(i,j) == matrixFromFile->GetElement(i,j)," matrix element (" << i << ", " << j << ") should be equal, but expected=" << matrix->GetElement(i,j) << ", and actual=" << matrixFromFile->GetElement(i, j));
      }
    }

  }
  catch (const itk::ExceptionObject&)
  {
    MITK_TEST_FAILED_MSG( << "Reading from file failed, which it should not.")
  }
  catch(...)
  {
    MITK_TEST_FAILED_MSG( << "Wrong exception (i.e. no itk:Exception) caught during read.")
  }

  // always end with this!
  MITK_TEST_END();
}
US_INITIALIZE_MODULE


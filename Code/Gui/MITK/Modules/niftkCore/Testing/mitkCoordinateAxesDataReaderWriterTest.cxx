/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <cstdlib>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <itksys/SystemTools.hxx>
#include <mitkTestingMacros.h>
#include <mitkTestingConfig.h>
#include <mitkDataStorage.h>
#include <mitkStandaloneDataStorage.h>
#include <mitkDataNode.h>
#include <mitkNifTKCoreObjectFactory.h>
#include <mitkBaseData.h>
#include <mitkBaseDataIOFactory.h>
#include <mitkCoordinateAxesData.h>
#include <mitkCoordinateAxesDataWriter.h>
#include <mitkIOUtil.h>

/**
 * \class
 * \brief Test class for mitk::CoordinateAxesData Input/Output.
 */
class CoordinateAxesDataReaderWriterTest
{

public:

  //-----------------------------------------------------------------------------
  void WriteThenReadFile(char* file)
  {
    std::string fileName = file;
    MITK_TEST_CONDITION_REQUIRED(fileName.size()>0,"check for filename")

    // Need to load images, specifically testing Reader and Writer factories, registered here.
    RegisterNifTKCoreObjectFactory();

    // Create matrix to store. The mitk::CoordinateAxesData only store the 3x3 rotation
    // matrix plus the offset. So the bottom row is always as per Identity matrix.
    vtkSmartPointer<vtkMatrix4x4> matrix = vtkMatrix4x4::New();
    matrix->Identity();
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 4; ++j)
      {
        matrix->SetElement(i, j, i+j);
      }
    }
    MITK_TEST_CONDITION_REQUIRED(matrix != NULL,"matrix instantiation")

    mitk::CoordinateAxesData::Pointer cad = mitk::CoordinateAxesData::New();
    cad->SetVtkMatrix(*matrix);

    mitk::CoordinateAxesDataWriter::Pointer writer = mitk::CoordinateAxesDataWriter::New();
    MITK_TEST_CONDITION_REQUIRED(writer.IsNotNull(),"writer instantiation")
    MITK_TEST_CONDITION_REQUIRED(writer->CanWriteBaseDataType(cad.GetPointer()),"writer can write data")

    writer->SetFileName(fileName);
    writer->DoWrite(cad.GetPointer());

    std::vector<mitk::BaseData::Pointer> matricesFromFile = mitk::BaseDataIO::LoadBaseDataFromFile(fileName, "", "", false );
    MITK_TEST_CONDITION_REQUIRED(matricesFromFile.size() > 0, "check if LoadBaseDataFromFile returned anything")

    mitk::BaseData* baseData = matricesFromFile.at(0);
    mitk::CoordinateAxesData* cadFromFile = dynamic_cast<mitk::CoordinateAxesData*>(baseData);
    MITK_TEST_CONDITION_REQUIRED(cadFromFile != NULL,"check if reader returned null")

    vtkSmartPointer<vtkMatrix4x4> matrixFromFile = vtkMatrix4x4::New();
    cadFromFile->GetVtkMatrix(*matrixFromFile);

    for (int i = 0; i < 4; ++i)
    {
      for (int j = 0; j < 4; ++j)
      {
        MITK_TEST_CONDITION_REQUIRED(matrix->GetElement(i,j) == matrixFromFile->GetElement(i,j)," matrix element (" << i << ", " << j << ") should be equal, but expected=" << matrix->GetElement(i,j) << ", and actual=" << matrixFromFile->GetElement(i, j));
      }
    }
  }
};

/**
 * Basic test harness for mitkMIDASImageUtilsTest.
 */
int mitkCoordinateAxesDataReaderWriterTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkCoordinateAxesDataReaderWriterTest");
  MITK_TEST_CONDITION_REQUIRED(argc>1,"check for filename argument")

  CoordinateAxesDataReaderWriterTest *testClass = new CoordinateAxesDataReaderWriterTest();
  testClass->WriteThenReadFile(argv[1]);
  delete testClass;

  MITK_TEST_END();
}



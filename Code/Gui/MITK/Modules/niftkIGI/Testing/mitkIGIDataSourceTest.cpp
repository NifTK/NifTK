/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : $Author: mjc $

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include <cstdlib>
#include <mitkTestingMacros.h>
#include "mitkIGIDataSource.h"
#include "mitkIGIDataType.h"

namespace mitk
{
class TestIGIDataSource : public IGIDataSource
{
public:
  mitkClassMacro(TestIGIDataSource, IGIDataSource);
  itkNewMacro(TestIGIDataSource);
  virtual void Initialize(){};
  mitk::IGIDataType::Pointer TestRequestData(igtlUint64 requestedTimeStamp)
  {
    return IGIDataSource::RequestData(requestedTimeStamp);
  }
protected:
  TestIGIDataSource() {};
  virtual ~TestIGIDataSource() {};
  virtual bool CanHandleData(mitk::IGIDataType* data) const { return true; }
};

} // end namespace

int mitkIGIDataSourceTest(int /*argc*/, char* /*argv*/[])
{

  mitk::TestIGIDataSource::Pointer dataSource = mitk::TestIGIDataSource::New();

  dataSource->SetIdentifier(1);
  MITK_TEST_CONDITION_REQUIRED(dataSource->GetIdentifier() == 1, ".. Testing Setter/Getter Identifier");

  dataSource->SetName("hello");
  MITK_TEST_CONDITION_REQUIRED(dataSource->GetName() == "hello", ".. Testing Setter/Getter Name");

  dataSource->SetType("test");
  MITK_TEST_CONDITION_REQUIRED(dataSource->GetType() == "test", ".. Testing Setter/Getter Type");

  dataSource->SetDescription("matt");
  MITK_TEST_CONDITION_REQUIRED(dataSource->GetDescription() ==  "matt", ".. Testing Setter/Getter Description");

  dataSource->SetSavePrefix("cool");
  MITK_TEST_CONDITION_REQUIRED(dataSource->GetSavePrefix() == "cool", ".. Testing Setter/Getter save prefix");

  dataSource->SetSaveState(false);
  MITK_TEST_CONDITION_REQUIRED(dataSource->GetSavingMessages() == false, ".. Testing Setter/Getter save prefix");
  MITK_TEST_CONDITION_REQUIRED(dataSource->GetFrameRate() == 0, ".. Testing default frame rate is zero");
  MITK_TEST_CONDITION_REQUIRED(dataSource->GetBufferSize() == 0, ".. Testing default buffer size is zero");

  mitk::IGIDataType::Pointer data1 = mitk::IGIDataType::New();
  data1->SetTimeStampUint64(1);
  data1->SetDuration(10);

  dataSource->AddData(data1);

  MITK_TEST_CONDITION_REQUIRED(dataSource->GetBufferSize() == 1, ".. Testing added 1 frame");
  MITK_TEST_CONDITION_REQUIRED(dataSource->GetFrameRate() == 0, ".. Testing frame rate is zero when we only have 1 frame");
  MITK_TEST_CONDITION_REQUIRED(dataSource->GetFirstTimeStamp() == 1, ".. Testing first time stamp = 1");
  MITK_TEST_CONDITION_REQUIRED(dataSource->GetLastTimeStamp() == 1, ".. Testing last time stamp = 1");

  mitk::IGIDataType::Pointer data2 = mitk::IGIDataType::New();
  data2->SetTimeStampUint64(5);
  data2->SetDuration(10);

  dataSource->AddData(data2);

  MITK_TEST_CONDITION_REQUIRED(dataSource->GetBufferSize() == 2, ".. Testing added 2 frames");
  MITK_TEST_CONDITION_REQUIRED(dataSource->GetFrameRate() == 0, ".. Testing frame rate is zero when we only have 2 frames");
  MITK_TEST_CONDITION_REQUIRED(dataSource->GetFirstTimeStamp() == 1, ".. Testing first time stamp = 1");
  MITK_TEST_CONDITION_REQUIRED(dataSource->GetLastTimeStamp() == 5, ".. Testing last time stamp = 5");

  mitk::IGIDataType::Pointer test = dataSource->TestRequestData(4);
  MITK_TEST_CONDITION_REQUIRED(test == data2, ".. Testing request nearest to second data point");

  test = dataSource->TestRequestData(1);
  MITK_TEST_CONDITION_REQUIRED(test == data1, ".. Testing request nearest to first data point");

  test = dataSource->TestRequestData(6);
  dataSource->SetTimeStampTolerance(5);

  MITK_TEST_CONDITION_REQUIRED(test == data2, ".. Testing request nearest to second data point, but after all data");
  MITK_TEST_CONDITION_REQUIRED(dataSource->IsCurrentWithinTimeTolerance() == true, ".. Testing that this frame is valid");

  test = dataSource->TestRequestData(16);
  MITK_TEST_CONDITION_REQUIRED(test == data2, ".. Testing request nearest to second data point, but after all data");
  MITK_TEST_CONDITION_REQUIRED(dataSource->IsCurrentWithinTimeTolerance() == false, ".. Testing that this frame is invalid");

  dataSource->CleanBuffer();
  MITK_TEST_CONDITION_REQUIRED(dataSource->GetBufferSize() == 1, ".. Calling clean, should only delete upto the current buffer.");

  dataSource->ClearBuffer();
  MITK_TEST_CONDITION_REQUIRED(dataSource->GetBufferSize() == 0, ".. Calling clear, should delete all buffer");

  for (int i = 1; i <= 20; i++)
  {
    mitk::IGIDataType::Pointer testData = mitk::IGIDataType::New();
    testData->SetTimeStampUint64(i+1);
    testData->SetDuration(10);
    dataSource->AddData(testData);

    float expectedFrameRate = 0;
    if (i >= 10)
    {
      expectedFrameRate = 1000;
    }
    dataSource->UpdateFrameRate();
    MITK_TEST_CONDITION_REQUIRED(dataSource->GetFrameRate() == expectedFrameRate, "... Testing expected frame rate, i=" << i << ", exp=" << expectedFrameRate << ", actual=" << dataSource->GetFrameRate());
  }
  return EXIT_SUCCESS;
}

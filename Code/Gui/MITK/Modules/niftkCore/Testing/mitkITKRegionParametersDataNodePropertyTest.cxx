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
#include <mitkVector.h>
#include <mitkTestingMacros.h>
#include <mitkITKRegionParametersDataNodeProperty.h>
#include <math.h>
#include <iostream>
#include <stdlib.h>

/**
 * \brief Test class for mitkITKRegionParametersDataNodePropertyTest
 */
class mitkITKRegionParametersDataNodePropertyTestClass
{

public:

  static void TestInstantiation()
  {
    MITK_TEST_OUTPUT(<< "Starting TestInstantiation...");

    mitk::ITKRegionParametersDataNodeProperty::Pointer prop1 = mitk::ITKRegionParametersDataNodeProperty::New();
    MITK_TEST_CONDITION_REQUIRED(prop1.IsNotNull(),"Testing instantiation with constructor 1.");

    MITK_TEST_OUTPUT(<< "Finished TestInstantiation...");
  }

  static void TestIdentity()
  {
    MITK_TEST_OUTPUT(<< "Starting TestIdentity...");

    mitk::ITKRegionParametersDataNodeProperty::Pointer prop1 = mitk::ITKRegionParametersDataNodeProperty::New();
    mitk::ITKRegionParametersDataNodeProperty::ParametersType params = prop1->GetITKRegionParameters();

    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(params[0], 0),".. Testing params[0]==0");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(params[1], 0),".. Testing params[1]==0");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(params[2], 0),".. Testing params[2]==0");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(params[3], 0),".. Testing params[3]==0");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(params[4], 0),".. Testing params[4]==0");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(params[5], 0),".. Testing params[5]==0");

    bool valid = prop1->IsValid();
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(valid, false),".. Testing valid==false");

    MITK_TEST_OUTPUT(<< "Finished TestIdentity...");
  }

  static void TestSetGet()
  {
    MITK_TEST_OUTPUT(<< "Starting TestSetGet...");

    mitk::ITKRegionParametersDataNodeProperty::Pointer prop1 = mitk::ITKRegionParametersDataNodeProperty::New();

    std::vector<int> params;
    params.push_back(1);
    params.push_back(3);
    params.push_back(5);
    params.push_back(4);
    params.push_back(2);
    params.push_back(6);

    prop1->SetITKRegionParameters(params);
    prop1->SetValid(true);

    std::vector<int> paramsOut = prop1->GetITKRegionParameters();
    bool validOut = prop1->IsValid();

    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(paramsOut[0], 1),".. Testing paramsOut[0]==1");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(paramsOut[1], 3),".. Testing paramsOut[1]==3");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(paramsOut[2], 5),".. Testing paramsOut[2]==5");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(paramsOut[3], 4),".. Testing paramsOut[3]==4");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(paramsOut[4], 2),".. Testing paramsOut[4]==2");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(paramsOut[5], 6),".. Testing paramsOut[5]==6");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(validOut, true),".. Testing validOut==true");

    MITK_TEST_OUTPUT(<< "Finished TestSetGet...");
  }
};

/**
 * Basic test harness for mitkITKRegionParametersDataNodeProperty.h
 */
int mitkITKRegionParametersDataNodePropertyTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkITKRegionParametersDataNodePropertyTest")

  mitkITKRegionParametersDataNodePropertyTestClass::TestInstantiation();
  mitkITKRegionParametersDataNodePropertyTestClass::TestIdentity();
  mitkITKRegionParametersDataNodePropertyTestClass::TestSetGet();

  MITK_TEST_END()
}


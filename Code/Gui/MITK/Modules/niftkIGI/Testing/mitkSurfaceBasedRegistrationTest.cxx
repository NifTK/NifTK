/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <mitkTestingMacros.h>
#include <mitkSurfaceBasedRegistration.h>

namespace mitk
{
class TestSurfaceBasedRegistration : public SurfaceBasedRegistration
{
public:
  mitkClassMacro(TestSurfaceBasedRegistration, SurfaceBasedRegistration);
  itkNewMacro(TestSurfaceBasedRegistration);
  virtual void Initialize(){};
protected:
  virtual ~TestSurfaceBasedRegistration() {};
};

} // end namespace

int mitkSurfaceBasedRegistrationTest(int /*argc*/, char* /*argv*/[])
{

  mitk::TestSurfaceBasedRegistration::Pointer registerer = mitk::TestSurfaceBasedRegistration::New();

  //tests
  //load fixed PointSet and fixed surface, and moving surface
  //register for both conditions, 
  //MITK_TEST_CONDITION_REQUIRED(registerGetTrandform() == 1, ".. Testing point to surface");
  //MITK_TEST_CONDITION_REQUIRED(registerGetTrandform() == 1, ".. Testing surface to surface");
  //Set rigid, non rigid, 
  //Set number of iterations, 
  //Set maximum number of points

  return EXIT_SUCCESS;
}

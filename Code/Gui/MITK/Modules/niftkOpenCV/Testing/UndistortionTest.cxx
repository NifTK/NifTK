/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <mitkTestingMacros.h>
#include <mitkCameraIntrinsicsProperty.h>
#include <mitkProperties.h>
#include <mitkPointSet.h>
#include <CameraCalibration/Undistortion.h>
#include <Conversion/ImageConversion.h>


//-----------------------------------------------------------------------------
static void TestErrorConditions()
{
  // calling constructors will null input does not make sense, so we expect an exception.
  try
  {
    niftk::Undistortion*  undist = new niftk::Undistortion(mitk::DataNode::Pointer());
    MITK_TEST_CONDITION(!"No exception thrown", "Undistortion constructor: Exception on null input node");
  }
  catch (const std::runtime_error& e)
  {
    MITK_TEST_CONDITION("Threw and caught correct exception", "Undistortion constructor: Exception on null input node");
  }
  catch (...)
  {
    MITK_TEST_CONDITION(!"Threw wrong exception", "Undistortion constructor: Exception on null input node");
  }

  try
  {
    niftk::Undistortion*  undist = new niftk::Undistortion(mitk::Image::Pointer());
    MITK_TEST_CONDITION(!"No exception thrown", "Undistortion constructor: Exception on null input image");
  }
  catch (const std::runtime_error& e)
  {
    MITK_TEST_CONDITION("Threw and caught correct exception", "Undistortion constructor: Exception on null input image");
  }
  catch (...)
  {
    MITK_TEST_CONDITION(!"Threw wrong exception", "Undistortion constructor: Exception on null input image");
  }


  // check that there is no exception if we pass in non-null input.
  try
  {
    niftk::Undistortion*  undist = new niftk::Undistortion(mitk::DataNode::New());
    MITK_TEST_CONDITION("No exception thrown", "Undistortion constructor: No exception on non-null input node");
  }
  catch (...)
  {
    MITK_TEST_CONDITION(!"Threw exception", "Undistortion constructor: No exception on non-null input node");
  }

  try
  {
    niftk::Undistortion*  undist = new niftk::Undistortion(mitk::Image::New());
    MITK_TEST_CONDITION("No exception thrown", "Undistortion constructor: No exception on non-null input image");
  }
  catch (...)
  {
    MITK_TEST_CONDITION(!"Threw exception", "Undistortion constructor: No exception on non-null input image");
  }


  // exception on null output.
  {
    niftk::Undistortion*  undist = new niftk::Undistortion(mitk::DataNode::New());
    try
    {
      undist->Run(mitk::DataNode::Pointer());
      MITK_TEST_CONDITION(!"No exception thrown", "Undistortion(node) Run(node): Exception on null output node");
    }
    catch (const std::runtime_error& e)
    {
      MITK_TEST_CONDITION("Threw and caught correct exception", "Undistortion(node) Run(node): Exception on null output node");
    }
    catch (...)
    {
      MITK_TEST_CONDITION(!"Threw wrong exception", "Undistortion(node) Run(node): Exception on null output node");
    }

    try
    {
      undist->Run(mitk::Image::Pointer());
      MITK_TEST_CONDITION(!"No exception thrown", "Undistortion(node) Run(image): Exception on null output image");
    }
    catch (const std::runtime_error& e)
    {
      MITK_TEST_CONDITION("Threw and caught correct exception", "Undistortion(node) Run(image): Exception on null output image");
    }
    catch (...)
    {
      MITK_TEST_CONDITION(!"Threw wrong exception", "Undistortion(node) Run(image): Exception on null output image");
    }

    delete undist;
  }

  {
    niftk::Undistortion*  undist = new niftk::Undistortion(mitk::Image::New());
    try
    {
      undist->Run(mitk::DataNode::Pointer());
      MITK_TEST_CONDITION(!"No exception thrown", "Undistortion(image) Run(node): Exception on null output node");
    }
    catch (const std::runtime_error& e)
    {
      MITK_TEST_CONDITION("Threw and caught correct exception", "Undistortion(image) Run(node): Exception on null output node");
    }
    catch (...)
    {
      MITK_TEST_CONDITION(!"Threw wrong exception", "Undistortion(image) Run(node): Exception on null output node");
    }

    try
    {
      undist->Run(mitk::Image::Pointer());
      MITK_TEST_CONDITION(!"No exception thrown", "Undistortion(image) Run(image): Exception on null output image");
    }
    catch (const std::runtime_error& e)
    {
      MITK_TEST_CONDITION("Threw and caught correct exception", "Undistortion(image) Run(image): Exception on null output image");
    }
    catch (...)
    {
      MITK_TEST_CONDITION(!"Threw wrong exception", "Undistortion(image) Run(image): Exception on null output image");
    }

    delete undist;
  }

  // input node without image attached --> exception on Run()
  {
    mitk::DataNode::Pointer   input  = mitk::DataNode::New();
    mitk::DataNode::Pointer   output = mitk::DataNode::New();
    niftk::Undistortion*    undist = new niftk::Undistortion(input);

    try
    {
      undist->Run(output);
      MITK_TEST_CONDITION(!"No exception thrown", "Undistortion(node) Run: Exception on empty node during Run()");
    }
    catch (const std::exception& e)
    {
      MITK_TEST_CONDITION("Threw and caught correct exception", "Undistortion(node) Run: Exception on empty node during Run()");
    }
    catch (...)
    {
      MITK_TEST_CONDITION(!"Threw wrong exception", "Undistortion(node) Run: Exception on empty node during Run()");
    }

    delete undist;
  }

  // input node with non-image data attached --> exception on Run()
  {
    mitk::DataNode::Pointer   output = mitk::DataNode::New();
    mitk::DataNode::Pointer   input = mitk::DataNode::New();
    input->SetData(mitk::PointSet::New());

    niftk::Undistortion*  undist = new niftk::Undistortion(input);

    try
    {
      undist->Run(output);
      MITK_TEST_CONDITION(!"No exception thrown", "Undistortion(node) Run: Exception on non-image input");
    }
    catch (const std::exception& e)
    {
      MITK_TEST_CONDITION("Threw and caught correct exception", "Undistortion(node) Run: Exception on non-image input");
    }
    catch (...)
    {
      MITK_TEST_CONDITION(!"Threw wrong exception", "Undistortion(node) Run: Exception on non-image input");
    }

    delete undist;
  }

  // input node with with zero-sized image during Run() --> exception on Run()
  {
    mitk::DataNode::Pointer   output = mitk::DataNode::New();
    mitk::DataNode::Pointer   input = mitk::DataNode::New();
    input->SetData(mitk::Image::New());

    niftk::Undistortion*  undist = new niftk::Undistortion(input);

    try
    {
      undist->Run(output);
      MITK_TEST_CONDITION(!"No exception thrown", "Undistortion(node) Run: Exception on zero input size");
    }
    catch (const std::exception& e)
    {
      MITK_TEST_CONDITION("Threw and caught correct exception", "Undistortion(node) Run: Exception on zero input size");
    }
    catch (...)
    {
      MITK_TEST_CONDITION(!"Threw wrong exception", "Undistortion(node) Run: Exception on zero input size");
    }

    delete undist;
  }

  // input image with zero size --> exception on Run()
  {
    mitk::DataNode::Pointer   output = mitk::DataNode::New();
    mitk::Image::Pointer      input = mitk::Image::New();

    niftk::Undistortion*  undist = new niftk::Undistortion(input);

    try
    {
      undist->Run(output);
      MITK_TEST_CONDITION(!"No exception thrown", "Undistortion(image) Run: Exception on zero input size");
    }
    catch (const std::exception& e)
    {
      MITK_TEST_CONDITION("Threw and caught correct exception", "Undistortion(image) Run: Exception on zero input size");
    }
    catch (...)
    {
      MITK_TEST_CONDITION(!"Threw wrong exception", "Undistortion(image) Run: Exception on zero input size");
    }

    delete undist;
  }

  // input node with calib prop of the right name but wrong type --> exception
  {
    IplImage* temp = cvCreateImage(cvSize(10, 10), IPL_DEPTH_8U, 4);
    mitk::Image::Pointer img = niftk::CreateMitkImage(temp);
    cvReleaseImage(&temp);

    mitk::DataNode::Pointer   output = mitk::DataNode::New();
    mitk::DataNode::Pointer   input = mitk::DataNode::New();
    input->SetData(img);
    input->SetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName, mitk::BoolProperty::New(true));

    niftk::Undistortion*  undist = new niftk::Undistortion(input);

    try
    {
      undist->Run(output);
      MITK_TEST_CONDITION(!"No exception thrown", "Undistortion(node) Run: Exception on invalid calibration property");
    }
    catch (const std::exception& e)
    {
      MITK_TEST_CONDITION("Threw and caught correct exception", "Undistortion(node) Run: Exception on invalid calibration property");
    }
    catch (...)
    {
      MITK_TEST_CONDITION(!"Threw wrong exception", "Undistortion(node) Run: Exception on invalid calibration property");
    }

    delete undist;
  }

  // input image with calib prop of the right name but wrong type --> exception
  {
    IplImage* temp = cvCreateImage(cvSize(10, 10), IPL_DEPTH_8U, 4);
    mitk::Image::Pointer input = niftk::CreateMitkImage(temp);
    cvReleaseImage(&temp);
    input->SetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName, mitk::BoolProperty::New(true));

    mitk::DataNode::Pointer   output = mitk::DataNode::New();

    niftk::Undistortion*  undist = new niftk::Undistortion(input);

    try
    {
      undist->Run(output);
      MITK_TEST_CONDITION(!"No exception thrown", "Undistortion(image) Run: Exception on invalid calibration property");
    }
    catch (const std::exception& e)
    {
      MITK_TEST_CONDITION("Threw and caught correct exception", "Undistortion(image) Run: Exception on invalid calibration property");
    }
    catch (...)
    {
      MITK_TEST_CONDITION(!"Threw wrong exception", "Undistortion(image) Run: Exception on invalid calibration property");
    }

    delete undist;
  }

  // input node without calib properties but with image (without calib props as well) --> exception
  {
    IplImage* temp = cvCreateImage(cvSize(10, 10), IPL_DEPTH_8U, 4);
    mitk::Image::Pointer img = niftk::CreateMitkImage(temp);
    cvReleaseImage(&temp);

    mitk::DataNode::Pointer   output = mitk::DataNode::New();
    mitk::DataNode::Pointer   input = mitk::DataNode::New();
    input->SetData(img);

    niftk::Undistortion*  undist = new niftk::Undistortion(input);

    try
    {
      undist->Run(input);
      MITK_TEST_CONDITION(!"No exception thrown", "Undistortion(node) Run: Exception on missing calibration property");
    }
    catch (const std::exception& e)
    {
      MITK_TEST_CONDITION("Threw and caught correct exception", "Undistortion(node) Run: Exception on missing calibration property");
    }
    catch (...)
    {
      MITK_TEST_CONDITION(!"Threw wrong exception", "Undistortion(node) Run: Exception on missing calibration property");
    }

    delete undist;
  }

  // input node with calib, but attached image without --> no exception
  {
    mitk::CameraIntrinsics::Pointer   cam = mitk::CameraIntrinsics::New();
    // values dont matter
    cam->SetFocalLength(1, 2);
    cam->SetPrincipalPoint(3, 4);
    cam->SetDistorsionCoeffs(5, 6, 7, 8);

    IplImage* temp = cvCreateImage(cvSize(10, 10), IPL_DEPTH_8U, 4);
    mitk::Image::Pointer img = niftk::CreateMitkImage(temp);
    cvReleaseImage(&temp);

    mitk::DataNode::Pointer   output = mitk::DataNode::New();
    mitk::DataNode::Pointer   input = mitk::DataNode::New();
    input->SetData(img);
    input->SetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName, mitk::CameraIntrinsicsProperty::New(cam));

    niftk::Undistortion*  undist = new niftk::Undistortion(input);

    try
    {
      undist->Run(output);
      MITK_TEST_CONDITION("No exception thrown", "Undistortion(node) Run: No exception for correct calibration property on node");
    }
    catch (...)
    {
      MITK_TEST_CONDITION(!"Threw exception", "Undistortion(node) Run: No exception for correct calibration property on node");
    }

    delete undist;
  }

  // input node without calib, but attached image with props --> no exception
  {
    mitk::CameraIntrinsics::Pointer   cam = mitk::CameraIntrinsics::New();
    cam->SetFocalLength(1, 2);
    cam->SetPrincipalPoint(3, 4);
    cam->SetDistorsionCoeffs(5, 6, 7, 8);

    IplImage* temp = cvCreateImage(cvSize(10, 10), IPL_DEPTH_8U, 4);
    mitk::Image::Pointer img = niftk::CreateMitkImage(temp);
    cvReleaseImage(&temp);
    img->SetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName, mitk::CameraIntrinsicsProperty::New(cam));

    mitk::DataNode::Pointer   output = mitk::DataNode::New();
    mitk::DataNode::Pointer   input = mitk::DataNode::New();
    input->SetData(img);

    niftk::Undistortion*  undist = new niftk::Undistortion(input);

    try
    {
      undist->Run(output);
      MITK_TEST_CONDITION("No exception thrown", "Undistortion(node) Run: No exception for correct calibration property on image");
    }
    catch (...)
    {
      MITK_TEST_CONDITION(!"Threw exception", "Undistortion(node) Run: No exception for correct calibration property on image");
    }

    delete undist;
  }

  // input node with calib, attached image with same calib. --> no exception
  {
    mitk::CameraIntrinsics::Pointer   cam = mitk::CameraIntrinsics::New();
    cam->SetFocalLength(1, 2);
    cam->SetPrincipalPoint(3, 4);
    cam->SetDistorsionCoeffs(5, 6, 7, 8);

    IplImage* temp = cvCreateImage(cvSize(10, 10), IPL_DEPTH_8U, 4);
    mitk::Image::Pointer img = niftk::CreateMitkImage(temp);
    cvReleaseImage(&temp);
    img->SetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName, mitk::CameraIntrinsicsProperty::New(cam));

    mitk::DataNode::Pointer   output = mitk::DataNode::New();
    mitk::DataNode::Pointer   input = mitk::DataNode::New();
    input->SetData(img);
    input->SetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName, mitk::CameraIntrinsicsProperty::New(cam));

    niftk::Undistortion*  undist = new niftk::Undistortion(input);

    try
    {
      undist->Run(output);
      MITK_TEST_CONDITION("No exception thrown", "Undistortion(node) Run: No exception for correct calibration property on both node and image");
    }
    catch (...)
    {
      MITK_TEST_CONDITION(!"Threw exception", "Undistortion(node) Run: No exception for correct calibration property on both node and image");
    }

    delete undist;
  }


  // FIXME: input node with calib, attached image with different calib. --> exception

  // input image and output image have different size.
  {
    IplImage* temp = cvCreateImage(cvSize(10, 10), IPL_DEPTH_8U, 4);
    mitk::Image::Pointer inputImage = niftk::CreateMitkImage(temp);
    cvReleaseImage(&temp);

    mitk::CameraIntrinsics::Pointer   cam = mitk::CameraIntrinsics::New();
    // values dont matter
    cam->SetFocalLength(1, 2);
    cam->SetPrincipalPoint(3, 4);
    cam->SetDistorsionCoeffs(5, 6, 7, 8);
    inputImage->SetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName, mitk::CameraIntrinsicsProperty::New(cam));

    temp = cvCreateImage(cvSize(11, 11), IPL_DEPTH_8U, 4);
    mitk::Image::Pointer outputImage = niftk::CreateMitkImage(temp);
    cvReleaseImage(&temp);

    niftk::Undistortion*  undist = new niftk::Undistortion(inputImage);

    try
    {
      undist->Run(outputImage);
      MITK_TEST_CONDITION(!"No exception thrown", "Undistortion Run(image): Exception on mismatched input/output image size");
    }
    catch (const std::runtime_error& e)
    {
      MITK_TEST_CONDITION("Threw and caught correct exception", "Undistortion Run(image): Exception on mismatched input/output image size");
    }
    catch (...)
    {
      MITK_TEST_CONDITION(!"Threw wrong exception", "Undistortion Run(image): Exception on mismatched input/output image size");
    }

    delete undist;
  }

  // FIXME: input image and output node with image of different size
  // FIXME: data-node with a non-zero-sized image attached, node ouput without image --> no exception
  // FIXME: data-node with a non-zero-sized image attached, node ouput with wrong-sized image --> no exception
  
  // input and output image are the same instance --> exception
  {
    IplImage* temp = cvCreateImage(cvSize(10, 10), IPL_DEPTH_8U, 4);
    mitk::Image::Pointer inputImage = niftk::CreateMitkImage(temp);
    cvReleaseImage(&temp);

    mitk::CameraIntrinsics::Pointer   cam = mitk::CameraIntrinsics::New();
    cam->SetFocalLength(1, 2);
    cam->SetPrincipalPoint(3, 4);
    cam->SetDistorsionCoeffs(5, 6, 7, 8);
    inputImage->SetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName, mitk::CameraIntrinsicsProperty::New(cam));

    niftk::Undistortion*  undist = new niftk::Undistortion(inputImage);

    try
    {
      undist->Run(inputImage);
      MITK_TEST_CONDITION(!"No exception thrown", "Undistortion(image) Run(image): Exception if input and output is same instance");
    }
    catch (const std::runtime_error& e)
    {
      MITK_TEST_CONDITION("Threw and caught correct exception", "Undistortion(image) Run(image): Exception if input and output is same instance");
    }
    catch (...)
    {
      MITK_TEST_CONDITION(!"Threw wrong exception", "Undistortion(image) Run(image): Exception if input and output is same instance");
    }

    delete undist;
  }

  // input and output node are the same instance --> exception
  {
    IplImage* temp = cvCreateImage(cvSize(10, 10), IPL_DEPTH_8U, 4);
    mitk::Image::Pointer inputImage = niftk::CreateMitkImage(temp);
    cvReleaseImage(&temp);

    mitk::CameraIntrinsics::Pointer   cam = mitk::CameraIntrinsics::New();
    cam->SetFocalLength(1, 2);
    cam->SetPrincipalPoint(3, 4);
    cam->SetDistorsionCoeffs(5, 6, 7, 8);
    inputImage->SetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName, mitk::CameraIntrinsicsProperty::New(cam));

    mitk::DataNode::Pointer   input = mitk::DataNode::New();
    input->SetData(inputImage);

    niftk::Undistortion*  undist = new niftk::Undistortion(input);

    try
    {
      undist->Run(input);
      MITK_TEST_CONDITION(!"No exception thrown", "Undistortion(node) Run(node): Exception if input and output is same instance");
    }
    catch (const std::runtime_error& e)
    {
      MITK_TEST_CONDITION("Threw and caught correct exception", "Undistortion(node) Run(node): Exception if input and output is same instance");
    }
    catch (...)
    {
      MITK_TEST_CONDITION(!"Threw wrong exception", "Undistortion(node) Run(node): Exception if input and output is same instance");
    }

    delete undist;
  }
}


//-----------------------------------------------------------------------------
static void TestOutput()
{
  // FIXME: check that output has input calib
  // FIXME: check that output has same size of input(-node)
  // FIXME: init with calib 1, run, change calib to 2, run again(), check that output has calib 2.
}


//-----------------------------------------------------------------------------
int UndistortionTest(int /*argc*/, char* /*argv*/[])
{
  MITK_TEST_BEGIN("UndistortionTest");
  TestErrorConditions();
  TestOutput();
  MITK_TEST_END();

  return EXIT_SUCCESS;
}

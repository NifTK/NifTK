/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "SurfaceReconstruction.h"
#include "SequentialCpuQds.h"
#include <opencv2/core/core_c.h>
#include <mitkImageReadAccessor.h>
#include <mitkCameraIntrinsicsProperty.h>
#include "../Conversion/ImageConversion.h"
#include <mitkPointSet.h>


namespace niftk 
{


const char*    SurfaceReconstruction::s_ImageIsUndistortedPropertyName      = "niftk.ImageIsUndistorted";
const char*    SurfaceReconstruction::s_ImageIsRectifiedPropertyName        = "niftk.ImageIsRectified";
const char*    SurfaceReconstruction::s_CameraCalibrationPropertyName       = "niftk.CameraCalibration";
const char*    SurfaceReconstruction::s_StereoRigTransformationPropertyName = "niftk.StereoRigTransformation";


//-----------------------------------------------------------------------------
SurfaceReconstruction::SurfaceReconstruction()
  : m_SequentialCpuQds(0)
{

}


//-----------------------------------------------------------------------------
SurfaceReconstruction::~SurfaceReconstruction()
{
  delete m_SequentialCpuQds;
}


//-----------------------------------------------------------------------------
void SurfaceReconstruction::Run(const mitk::DataStorage::Pointer dataStorage,
                                mitk::DataNode::Pointer outputNode,
                                const mitk::Image::Pointer image1,
                                const mitk::Image::Pointer image2,
                                Method method,
                                OutputType outputtype)
{
  // sanity check
  assert(dataStorage.IsNotNull());
  assert(image1.IsNotNull());
  assert(image2.IsNotNull());

  unsigned int width  = image1->GetDimension(0);
  unsigned int height = image1->GetDimension(1);

  // for current methods, both left and right have to have the same size
  if (image2->GetDimension(0) != width)
  {
    throw std::runtime_error("Left and right image width are different");
  }
  if (image2->GetDimension(1) != height)
  {
    throw std::runtime_error("Left and right image height are different");
  }
  // we dont really care here whether the image has a z dimension or not
  // but for debugging purposes might as well check
  assert(image1->GetDimension(2) == 1);
  assert(image2->GetDimension(2) == 1);


  try
  {
    mitk::ImageReadAccessor  leftReadAccess(image1);
    mitk::ImageReadAccessor  rightReadAccess(image2);

    const void* leftPtr = leftReadAccess.GetData();
    const void* rightPtr = rightReadAccess.GetData();

    int numComponents = image1->GetPixelType().GetNumberOfComponents();
    assert((int)(image2->GetPixelType().GetNumberOfComponents()) == numComponents);

    // mitk images are tightly packed (i hope)
    int bytesPerRow = width * numComponents * (image1->GetPixelType().GetBitsPerComponent() / 8);

    IplImage  leftIpl;
    cvInitImageHeader(&leftIpl, cvSize(width, height), IPL_DEPTH_8U, numComponents);
    cvSetData(&leftIpl, const_cast<void*>(leftPtr), bytesPerRow);
    IplImage  rightIpl;
    cvInitImageHeader(&rightIpl, cvSize(width, height), IPL_DEPTH_8U, numComponents);
    cvSetData(&rightIpl, const_cast<void*>(rightPtr), bytesPerRow);


    switch (method)
    {
      case SEQUENTIAL_CPU:
      {
        if (m_SequentialCpuQds != 0)
        {
          // internal buffers of SeqQDS are fixed during construction
          // but our input images can vary in size
          if ((m_SequentialCpuQds->GetWidth()  != (int)width) ||
              (m_SequentialCpuQds->GetHeight() != (int)height))
          {
            // will be recreated below, with the correct dimensions
            delete m_SequentialCpuQds;
            m_SequentialCpuQds = 0;
          }
        }

        // may not have been created before
        // or may have been deleted above in the size check
        if (m_SequentialCpuQds == 0)
          m_SequentialCpuQds = new SequentialCpuQds(width, height);

        m_SequentialCpuQds->Process(&leftIpl, &rightIpl);

        // FIXME: when there are more methods implemented we should refactor this stuff!
        switch (outputtype)
        {
          case POINT_CLOUD:
          {
            // it's an error to request point cloud and not have a calibration!
            // FIXME: move this to the top of the method!
            mitk::BaseProperty::Pointer       cam1bp = image1->GetProperty(niftk::SurfaceReconstruction::s_CameraCalibrationPropertyName);
            mitk::BaseProperty::Pointer       cam2bp = image2->GetProperty(niftk::SurfaceReconstruction::s_CameraCalibrationPropertyName);
            if (cam1bp.IsNull() || cam2bp.IsNull())
            {
              throw std::runtime_error("Image has to have a calibration for point cloud output");
            }
            mitk::CameraIntrinsicsProperty::Pointer   cam1 = dynamic_cast<mitk::CameraIntrinsicsProperty*>(cam1bp.GetPointer());
            mitk::CameraIntrinsicsProperty::Pointer   cam2 = dynamic_cast<mitk::CameraIntrinsicsProperty*>(cam2bp.GetPointer());
            if (cam1.IsNull() || cam2.IsNull())
            {
              throw std::runtime_error("Image does not have a valid calibration which is required for point cloud output");
            }

            mitk::BaseProperty::Pointer   rigbp = image1->GetProperty(niftk::SurfaceReconstruction::s_StereoRigTransformationPropertyName);
            // FIXME: undecided whether both channels should have a stereo-rig transform, whether they need to match, or only one channel
            if (rigbp.IsNull())
            {
              rigbp = image2->GetProperty(niftk::SurfaceReconstruction::s_StereoRigTransformationPropertyName);
              if (rigbp.IsNull())
              {
                throw std::runtime_error("Images need a stereo-rig transformation for point cloud output");
              }
            }
            niftk::MatrixProperty::Pointer  stereorig = dynamic_cast<niftk::MatrixProperty*>(rigbp.GetPointer());
            if (stereorig.IsNull())
            {
              throw std::runtime_error("Images do not have a valid stereo-rig transformation which is required for point cloud output");
            }

            mitk::PointSet::Pointer   points = mitk::PointSet::New();

            for (unsigned int y = 0; y < height; ++y)
            {
              for (unsigned int x = 0; x < width; ++x)
              {
                CvPoint r = m_SequentialCpuQds->GetMatch(x, y);
                if (r.x != 0)
                {
                  BOOST_STATIC_ASSERT((sizeof(CvPoint3D32f) == 3 * sizeof(float)));
                  float  error = 0;
                  CvPoint3D32f p = niftk::triangulate(
                                       x,   y, cam1->GetValue(),
                                     r.x, r.y, cam2->GetValue(),
                                     stereorig->GetValue(),
                                     &error);
                  if (error < 0.1)
                  {
                    points->InsertPoint(y * width + x, mitk::PointSet::PointType(&p.x));
                  }
                }
              }
            }
            outputNode->SetData(points);
            break;
          }

          case DISPARITY_IMAGE:
          {
            IplImage* dispimg = m_SequentialCpuQds->CreateDisparityImage();
            mitk::Image::Pointer imgData4Node = CreateMitkImage(dispimg);
            cvReleaseImage(&dispimg);

            outputNode->SetData(imgData4Node);
            break;
          }
        } // end switch on output
      } // end sequential CPU
      break;
      default:
        throw std::logic_error("Method not implemented");
    } // end switch method
  }
  catch (const mitk::Exception& e)
  {
    throw std::runtime_error(std::string("Something went wrong with MITK bits: ") + e.what());
  }
}

} // end namespace

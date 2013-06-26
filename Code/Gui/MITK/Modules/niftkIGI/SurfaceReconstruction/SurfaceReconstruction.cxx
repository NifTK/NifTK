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
#include <mitkCoordinateAxesData.h>


namespace niftk 
{


const char*    SurfaceReconstruction::s_ImageIsRectifiedPropertyName        = "niftk.ImageIsRectified";
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
mitk::BaseData::Pointer SurfaceReconstruction::Run(ParamPacket params)
{
  return this->Run(params.image1, params.image2, params.method, params.outputtype, params.camnode, params.maxTriangulationError, params.minDepth, params.maxDepth);
}


//-----------------------------------------------------------------------------
mitk::BaseData::Pointer SurfaceReconstruction::Run(
                                const mitk::Image::Pointer image1,
                                const mitk::Image::Pointer image2,
                                Method method,
                                OutputType outputtype,
                                const mitk::DataNode::Pointer camnode,
                                float maxTriangulationError,
                                float minDepth,
                                float maxDepth)
{
  // sanity check
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

  // calibration properties needed for pointcloud output
  mitk::CameraIntrinsicsProperty::Pointer   camIntr1;
  mitk::CameraIntrinsicsProperty::Pointer   camIntr2;
  niftk::MatrixProperty::Pointer            stereoRig;

  // check this before we start wasting cpu cycles
  if (outputtype == POINT_CLOUD)
  {
    mitk::BaseProperty::Pointer       cam1bp = image1->GetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName);
    mitk::BaseProperty::Pointer       cam2bp = image2->GetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName);
    // it's an error to request point cloud and not have a calibration!
    if (cam1bp.IsNull() || cam2bp.IsNull())
    {
      throw std::runtime_error("Image has to have a calibration for point cloud output");
    }

    camIntr1 = dynamic_cast<mitk::CameraIntrinsicsProperty*>(cam1bp.GetPointer());
    camIntr2 = dynamic_cast<mitk::CameraIntrinsicsProperty*>(cam2bp.GetPointer());
    if (camIntr1.IsNull() || camIntr2.IsNull())
    {
      throw std::runtime_error("Image does not have a valid calibration which is required for point cloud output");
    }

    mitk::BaseProperty::Pointer   rigbp = image1->GetProperty(niftk::SurfaceReconstruction::s_StereoRigTransformationPropertyName);
    // undecided whether both channels should have a stereo-rig transform, whether they need to match, or only one channel
    if (rigbp.IsNull())
    {
      rigbp = image2->GetProperty(niftk::SurfaceReconstruction::s_StereoRigTransformationPropertyName);
      if (rigbp.IsNull())
      {
        throw std::runtime_error("Images need a stereo-rig transformation for point cloud output");
      }
    }
    stereoRig = dynamic_cast<niftk::MatrixProperty*>(rigbp.GetPointer());
    if (stereoRig.IsNull())
    {
      throw std::runtime_error("Images do not have a valid stereo-rig transformation which is required for point cloud output");
    }

  }


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

    // irrespective of which method we will be running below,
    // make sure sizes are correct. (no real need for that but it makes
    // handling below simpler.)
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

    QDSInterface*   methodImpl = 0;

    switch (method)
    {
      case SEQUENTIAL_CPU:
      {
        // may not have been created before
        // or may have been deleted above in the size check
        if (m_SequentialCpuQds == 0)
        {
          m_SequentialCpuQds = new SequentialCpuQds(width, height);
        }

        methodImpl = m_SequentialCpuQds;
        break;
      }

      default:
        throw std::logic_error("Method not implemented");
    } // end switch method


    methodImpl->Process(&leftIpl, &rightIpl);


    switch (outputtype)
    {
      case POINT_CLOUD:
      {
        mitk::PointSet::Pointer   points = mitk::PointSet::New();

        for (unsigned int y = 0; y < height; ++y)
        {
          for (unsigned int x = 0; x < width; ++x)
          {
            CvPoint r = methodImpl->GetMatch(x, y);
            if (r.x != 0)
            {
              BOOST_STATIC_ASSERT((sizeof(CvPoint3D32f) == 3 * sizeof(float)));
              float  error = 0;
              CvPoint3D32f p = niftk::triangulate(
                                    x,   y, camIntr1->GetValue(),
                                  r.x, r.y, camIntr2->GetValue(),
                                  stereoRig->GetValue(),
                                  &error);
              p.y = -p.y;
              p.z = -p.z;
              if (error < maxTriangulationError)
              {
                float   depth = std::sqrt((p.x * p.x) + (p.y * p.y) + (p.z * p.z));
                if (depth >= minDepth)
                {
                  if (depth <= maxDepth)
                  {
                    points->InsertPoint(y * width + x, mitk::PointSet::PointType(&p.x));
                  }
                }
              }
            }
          }
        }
        //outputNode->SetData(points);

        // if our camnode has mitk::CoordinateAxesData::Pointer then we use that.
        // otherwise we copy its geometry.
        if (camnode.IsNotNull())
        {
          mitk::BaseData::Pointer   camnodebasedata = camnode->GetData();
          if (camnodebasedata.IsNotNull())
          {
            // it's actually irrelevant what type the data is
            const mitk::Geometry3D::Pointer geom = camnodebasedata->GetGeometry();
            // is there no usable clone???
            points->GetGeometry()->SetSpacing(geom->GetSpacing());
            points->GetGeometry()->SetOrigin(geom->GetOrigin());
            points->GetGeometry()->SetIndexToWorldTransform(geom->GetIndexToWorldTransform());
            points->GetGeometry()->SetObjectToNodeTransform(geom->GetObjectToNodeTransform());
          }
        }

        return points.GetPointer();
      }

      case DISPARITY_IMAGE:
      {
        IplImage* dispimg = methodImpl->CreateDisparityImage();
        mitk::Image::Pointer imgData4Node = CreateMitkImage(dispimg);
        cvReleaseImage(&dispimg);

        // disparity image is in the view of the left eye.
        // copy the calibration properties to the output so that it's rendered properly with any overlaid geometry.
        mitk::BaseProperty::Pointer       cam1bp = image1->GetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName);
        if (cam1bp.IsNotNull())
        {
          imgData4Node->SetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName, cam1bp);
        }
        mitk::BaseProperty::Pointer       undist1bp = image1->GetProperty(niftk::Undistortion::s_ImageIsUndistortedPropertyName);
        if (undist1bp.IsNotNull())
        {
          imgData4Node->SetProperty(niftk::Undistortion::s_ImageIsUndistortedPropertyName, undist1bp);
        }

        //outputNode->SetData(imgData4Node);
        return imgData4Node.GetPointer();
      }
    } // end switch on output

  }
  catch (const mitk::Exception& e)
  {
    throw std::runtime_error(std::string("Something went wrong with MITK bits: ") + e.what());
  }

  return 0;
}

} // end namespace

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
#include <mitkCameraCalibrationFacade.h>
#ifdef _USE_PCL
#include <mitkPCLData.h>
#endif

namespace niftk 
{


struct MethodDescription
{
  const char*                           m_Name;
  const SurfaceReconstruction::Method   m_ID;
};

static const MethodDescription s_AvailableMethods[] =
{
  // name needs to be unique!
  {"Sequential Quasi-Dense CPU", SurfaceReconstruction::SEQUENTIAL_CPU}
};


//-----------------------------------------------------------------------------
bool SurfaceReconstruction::GetMethodDetails(int index, Method* id, std::string* friendlyname)
{
  if (index < 0)
    return false;
  if (index >= (sizeof(s_AvailableMethods) / sizeof(s_AvailableMethods[0])))
    return false;

  if (id != 0)
  {
    *id = s_AvailableMethods[index].m_ID;
  }
  if (friendlyname != 0)
  {
    *friendlyname = s_AvailableMethods[index].m_Name;
  }

  return true;
}


//-----------------------------------------------------------------------------
SurfaceReconstruction::Method SurfaceReconstruction::ParseMethodName(const std::string& friendlyname)
{
  // dumb sequential scan.
  for (int i = 0; i < (sizeof(s_AvailableMethods) / sizeof(s_AvailableMethods[0])); ++i)
  {
    if (s_AvailableMethods[i].m_Name == friendlyname)
      return s_AvailableMethods[i].m_ID;
  }

  throw std::runtime_error("SurfaceReconstruction::ParseMethodName: unknown method name");
}


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
  return this->Run(params.image1, params.image2, params.method, params.outputtype, params.camnode, params.maxTriangulationError, params.minDepth, params.maxDepth, params.bakeCameraTransform);
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
                                float maxDepth,
                                bool bakeCameraTransform)
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

#ifndef _USE_PCL
  if (outputtype == PCL_POINT_CLOUD)
  {
    throw std::logic_error("Cannot output PCL pointcloud, no PCL support compiled in");
  }
#endif

  // check this before we start wasting cpu cycles
  if ((outputtype == MITK_POINT_CLOUD) ||
      (outputtype == PCL_POINT_CLOUD))
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

    mitk::BaseProperty::Pointer   rigbp = image1->GetProperty(niftk::Undistortion::s_StereoRigTransformationPropertyName);
    // undecided whether both channels should have a stereo-rig transform, whether they need to match, or only one channel
    if (rigbp.IsNull())
    {
      rigbp = image2->GetProperty(niftk::Undistortion::s_StereoRigTransformationPropertyName);
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

    // after locking the images, grab the position of the camera.
    // this should keep them in sync better, if we are reconstructing a point cloud while the camera is moving.
    mitk::BaseGeometry::Pointer   camgeom;
    if (camnode.IsNotNull())
    {
      mitk::BaseData::Pointer   camnodebasedata = camnode->GetData();
      if (camnodebasedata.IsNotNull())
      {
        camgeom = dynamic_cast<mitk::BaseGeometry*>(camnodebasedata->GetGeometry()->Clone().GetPointer());
      }
    }
    mitk::BaseGeometry::Pointer   imggeom = dynamic_cast<mitk::BaseGeometry*>(image1->GetGeometry()->Clone().GetPointer());

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
      case MITK_POINT_CLOUD:
      case PCL_POINT_CLOUD:
      {
        cv::Point2d leftPixel;
        cv::Point2d rightPixel;
        cv::Mat left2right_rotation = cv::Mat(3, 3, CV_32F, (void*) &stereoRig->GetValue().GetVnlMatrix()(0, 0), sizeof(float) * 4);
        cv::Mat left2right_translation = cv::Mat(1, 3, CV_32F);
        left2right_translation.at<float>(0,0) = stereoRig->GetValue()[0][3];
        left2right_translation.at<float>(0,1) = stereoRig->GetValue()[1][3];
        left2right_translation.at<float>(0,2) = stereoRig->GetValue()[2][3];

        std::vector< cv::Point3d > outputOpenCVPoints;
        std::vector< std::pair<cv::Point2d, cv::Point2d> > inputUndistortedPoints;
#ifdef _USE_PCL
        std::vector<cv::Point3f>    pointcolours;
#endif

        // Get valid point pairs.
        for (unsigned int y = 0; y < height; ++y)
        {
          for (unsigned int x = 0; x < width; ++x)
          {
            CvPoint r = methodImpl->GetMatch(x, y);
            if (r.x != 0)
            {
              BOOST_STATIC_ASSERT((sizeof(CvPoint3D32f) == 3 * sizeof(float)));

              leftPixel.x = x;
              leftPixel.y = y;
              rightPixel.x = r.x;
              rightPixel.y = r.y;
              inputUndistortedPoints.push_back(std::pair<cv::Point2d, cv::Point2d>(leftPixel, rightPixel));
#ifdef _USE_PCL
              if (outputtype == PCL_POINT_CLOUD)
              {
                CvScalar rgba = cvGet2D(&leftIpl, y, x);
                pointcolours.push_back(cv::Point3f(rgba.val[0], rgba.val[1], rgba.val[2]));
              }
#endif
            }
          }
        }

        // Triangulate them all in one go.
        outputOpenCVPoints = mitk::TriangulatePointPairsUsingGeometry(
            inputUndistortedPoints,
            camIntr1->GetValue()->GetCameraMatrix(),
            camIntr2->GetValue()->GetCameraMatrix(),
            left2right_rotation,
            left2right_translation,
            maxTriangulationError,
            true
            );

        // Filter by depth.
        cv::Point3d p;
        mitk::Point3D outputPoint;
        mitk::PointSet::Pointer points = mitk::PointSet::New();
#ifdef _USE_PCL
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr  cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        mitk::PCLData::Pointer                  pcldata = mitk::PCLData::New();
        pcldata->SetCloud(cloud);
#endif

        assert(outputOpenCVPoints.size() == inputUndistortedPoints.size());
        for (unsigned int i = 0; i < outputOpenCVPoints.size(); i++)
        {
          p = outputOpenCVPoints[i];
          double depth = std::sqrt((p.x * p.x) + (p.y * p.y) + (p.z * p.z));
                                // FIXME: extra check temporarily disabled
          if (depth >= minDepth)// && p.z > minDepth)
          {
            if (depth <= maxDepth)
            {
              outputPoint[0] = p.x;
              outputPoint[1] = p.y;
              outputPoint[2] = p.z;
              if (outputtype == MITK_POINT_CLOUD)
              {
                points->InsertPoint(i,outputPoint);
              }
#ifdef _USE_PCL
              else
              if (outputtype == PCL_POINT_CLOUD)
              {
                pcl::PointXYZRGB  q(pointcolours[i].x, pointcolours[i].y, pointcolours[i].z);
                q.x = p.x;
                q.y = p.y;
                q.z = p.z;
                cloud->push_back(q);
              }
#endif
              else
                // should not happen!
                assert(false);
            }
          }
        }

        if (camgeom.IsNotNull())
        {
          points->GetGeometry()->SetSpacing(camgeom->GetSpacing());
          points->GetGeometry()->SetOrigin(camgeom->GetOrigin());
          points->GetGeometry()->SetIndexToWorldTransform(camgeom->GetIndexToWorldTransform());
#ifdef _USE_PCL
          pcldata->GetGeometry()->SetSpacing(camgeom->GetSpacing());
          pcldata->GetGeometry()->SetOrigin(camgeom->GetOrigin());
          pcldata->GetGeometry()->SetIndexToWorldTransform(camgeom->GetIndexToWorldTransform());
#endif

          if (bakeCameraTransform)
          {
            // use the transformation we just set to get the point in world coordinates.
            // then stuff it back in (and this means it would be transformed twice!).
            // then strip off the camera transformation.
            for (mitk::PointSet::PointsIterator i = points->Begin(); i != points->End(); ++i)
            {
              // we need the point in world-coordinates! i.e. take its index-to-world transformation into account.
              // so instead of i->Value() we go via GetPointIfExists(i->Id(), ...)
              mitk::PointSet::PointType p;
              bool pointexists = points->GetPointIfExists(i->Index(), &p);
              // sanity check
              assert(pointexists);

              // directly overwrite coordinates.
              // do not use SetPoint()! it will try to transform p back into the local coordinate system.
              i->Value() = p;
            }

            // camgeom has been cloned off the camera node above.
            // so while we might reset a shared instance of geometry, we are only sharing it with outself here.
            points->GetGeometry()->SetIdentity();
          }
        }

        if (outputtype == MITK_POINT_CLOUD)
          return points.GetPointer();
#ifdef _USE_PCL
        else
          return pcldata.GetPointer();
#endif
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
        // copy input geometry too. so that a field-dropped input image
        // doesnt cause the ouput to look squashed.
        imgData4Node->GetGeometry()->SetSpacing(imggeom->GetSpacing());
        imgData4Node->GetGeometry()->SetOrigin(imggeom->GetOrigin());
        imgData4Node->GetGeometry()->SetIndexToWorldTransform(imggeom->GetIndexToWorldTransform());

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

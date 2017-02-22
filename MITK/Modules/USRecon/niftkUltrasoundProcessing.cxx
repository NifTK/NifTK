/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include "niftkUltrasoundProcessing.h"
#include <mitkExceptionMacro.h>
#include <niftkOpenCVImageConversion.h>
#include <mitkOpenCVMaths.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>
#include <mitkImageAccessByItk.h>
#include <cv.h>

namespace niftk
{

//-----------------------------------------------------------------------------
template <typename TPixel1, unsigned int VImageDimension1,
          typename TPixel2, unsigned int VImageDimension2>
void ITKReconstructOneSlice(const itk::Image<TPixel1, VImageDimension1>* input,
                            itk::Image<TPixel2, VImageDimension2>* output)
{
  typedef typename itk::Image<TPixel1, VImageDimension1> ImageType1;
  typedef typename itk::Image<TPixel2, VImageDimension2> ImageType2;

  // Iterate through input, writing to output.
}


//-----------------------------------------------------------------------------
mitk::Image::Pointer DoUltrasoundReconstruction(const TrackedImageData& data,
                                                const vtkMatrix4x4& pixelToSensorTransform
                                                )
{
  MITK_INFO << "DoUltrasoundReconstruction: Doing Ultrasound Reconstruction with "
            << data.size() << " samples.";

  if (data.size() == 0)
  {
    mitkThrow() << "No reconstruction data provided.";
  }

  vtkSmartPointer<vtkMatrix4x4> trackingMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  trackingMatrix->Identity();

  vtkSmartPointer<vtkMatrix4x4> indexToWorld = vtkSmartPointer<vtkMatrix4x4>::New();
  indexToWorld->Identity();

  // Calculate size of bounding box.
  mitk::Point3D minCornerInMillimetres;
  mitk::Point3D maxCornerInMillimetres;

  for (unsigned int i = 0; i < data.size(); i++)
  {
    if (data[i].first.IsNull())
    {
      mitkThrow() << "Ultrasound image " << i << " is NULL?!?!?";
    }
    if (data[i].first->GetDimension() != 3)
    {
      mitkThrow() << "Ultrasound images should be 3D.";
    }
    if (data[i].first->GetDimensions()[2] != 1)
    {
      mitkThrow() << "Ultrasound images should be 3D, with 1 slice.";
    }

    niftk::CoordinateAxesData::Pointer trackingTransform = data[i].second;
    trackingTransform->GetVtkMatrix(*trackingMatrix);

    vtkMatrix4x4::Multiply4x4(trackingMatrix, &pixelToSensorTransform, indexToWorld);

    mitk::Image::Pointer trackedImage = data[i].first;
    trackedImage->GetGeometry()->SetIndexToWorldTransformByVtkMatrix(indexToWorld);

    // multiply min, max pixel index by indexToWorld
    // check for most negative and most positive x,y,z coordinate.
    // store in minCornerInMillimetres, maxCornerInMillimetres
  }

  unsigned int dim[3];
  dim[0] = 5; // put number of voxels in x
  dim[1] = 5; // put number of voxels in y
  dim[2] = 5; // put number of voxels in z

  mitk::Vector3D spacing;
  spacing[0] = 1; // put size of voxels in x in millimetres
  spacing[1] = 1; // put size of voxels in y in millimetres
  spacing[2] = 1; // put size of voxels in z in millimetres

  // See MITK docs about image origins.
  mitk::Point3D origin;
  origin[0] = minCornerInMillimetres[0]-(0.5 * spacing[0]); // put origin position in millimetres.
  origin[1] = minCornerInMillimetres[1]-(0.5 * spacing[1]); // put origin position in millimetres.
  origin[2] = minCornerInMillimetres[2]-(0.5 * spacing[2]); // put origin position in millimetres.

  mitk::PixelType pixelType = data[0].first->GetPixelType();

  mitk::Image::Pointer image3D = mitk::Image::New();
  image3D->Initialize(pixelType, 3, dim);
  image3D->SetSpacing(spacing);
  image3D->SetOrigin(origin);

  // Now iterate through each image/tracking, and put in volume.
  for (unsigned int i = 0; i < data.size(); i++)
  {
    mitk::Image::Pointer image2D = data[i].first;

    try
    {
      AccessTwoImagesFixedDimensionByItk(image2D.GetPointer(),
                                         image3D.GetPointer(),
                                         ITKReconstructOneSlice,
                                        3); // has to be 3D at this point.
    }
    catch (const mitk::AccessByItkException &e)
    {
      MITK_ERROR << "ITKReconstructOneSlice: AccessTwoImagesFixedDimensionByItk failed to reconstruct "
                 << " image data " << i << " due to."
                 << e.what() << std::endl;
    }
  }

  // And returns the image.
  return image3D;
}


//-----------------------------------------------------------------------------
cv::Point2d FindCircleInImage(const cv::Mat& image)
{
  cv::Point2d result;
  return result;
}


//-----------------------------------------------------------------------------
std::vector<double> DoUltrasoundCalibration(const std::vector<cv::Point2d>& points,
                                            const std::vector<cv::Matx44d>& matrices)
{
  // Feel free to simplify the method below to pass in cv::Mat here
  // for example, .... if that helps things along.

  std::vector<double> result;
  return result;
}


//-----------------------------------------------------------------------------
void DoUltrasoundCalibration(const TrackedImageData& data,
                             vtkMatrix4x4& pixelToMillimetreScale,
                             vtkMatrix4x4& imageToSensorTransform
                             )
{
  MITK_INFO << "DoUltrasoundCalibration: Doing Ultrasound Calibration with "
            << data.size() << " samples.";

  std::vector<cv::Point2d> points;
  std::vector<cv::Matx44d> matrices;

  // Extract all 2D centres of circles
  for (int i = 0; i < data.size(); i++)
  {
    // Feel free to use other OpenCV types.
    // These are just some of the examples in Niftk.

    cv::Mat tmpImage = niftk::MitkImageToOpenCVMat(data[i].first);
    cv::Point2d pixelLocation = niftk::FindCircleInImage(tmpImage);

    vtkSmartPointer<vtkMatrix4x4> vtkMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
    data[i].second->GetVtkMatrix(*vtkMatrix);

    cv::Matx44d trackingMatrix;
    mitk::CopyToOpenCVMatrix(*vtkMatrix, trackingMatrix);

    points.push_back(pixelLocation);
    matrices.push_back(trackingMatrix);
  }

  // Now do calibration.
  // Feel free to change return types.
  std::vector<double> parameters = DoUltrasoundCalibration(points, matrices);

  // Now copy into output VTK matrices
  pixelToMillimetreScale.Identity();
  pixelToMillimetreScale.SetElement(0, 0, 1 /* set scale here */);
  pixelToMillimetreScale.SetElement(1, 1, 1 /* set scale here */);

  imageToSensorTransform.Identity();
  /* set rigid body matrix here */
}

} // end namespace

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkStereoImageToModelMetric.h"
#include "mitkRegistrationHelper.h"
#include "mitkCameraCalibrationFacade.h"
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>

namespace mitk
{

//-----------------------------------------------------------------------------
StereoImageToModelMetric::StereoImageToModelMetric()
: m_PolyDataReader(NULL)
, m_InputLeftImage(NULL)
, m_InputRightImage(NULL)
, m_IntrinsicLeft(NULL)
, m_DistortionLeft(NULL)
, m_RotationLeft(NULL)
, m_TranslationLeft(NULL)
, m_IntrinsicRight(NULL)
, m_DistortionRight(NULL)
, m_RightToLeftRotation(NULL)
, m_RightToLeftTranslation(NULL)
, m_ModelPoints(NULL)
, m_ModelNormals(NULL)
, m_CameraNormal(NULL)
, m_OutputLeftImage(NULL)
, m_OutputRightImage(NULL)
, m_DrawOutput(false)
{
  m_Parameters.SetSize(6);
  m_Parameters.Fill(0);
}


//-----------------------------------------------------------------------------
StereoImageToModelMetric::~StereoImageToModelMetric()
{
  m_PolyDataReader->Delete();

  cvReleaseImage(&m_InputLeftImage);
  cvReleaseImage(&m_InputRightImage);

  cvReleaseMat(&m_IntrinsicLeft);
  cvReleaseMat(&m_DistortionLeft);
  cvReleaseMat(&m_RotationLeft);
  cvReleaseMat(&m_TranslationLeft);
  cvReleaseMat(&m_IntrinsicRight);
  cvReleaseMat(&m_DistortionRight);
  cvReleaseMat(&m_RightToLeftRotation);
  cvReleaseMat(&m_RightToLeftTranslation);
  cvReleaseMat(&m_ModelPoints);
  cvReleaseMat(&m_ModelNormals);
  cvReleaseMat(&m_CameraNormal);

  cvReleaseImage(&m_OutputLeftImage);
  cvReleaseImage(&m_OutputRightImage);
}


//-----------------------------------------------------------------------------
void StereoImageToModelMetric::Initialize()
{
  if (   m_Input3DModelFileName.size() == 0
      || m_InputLeftImageFileName.size() == 0
      || m_InputRightImageFileName.size() == 0
      || m_OutputLeftImageFileName.size() == 0
      || m_OutputRightImageFileName.size() == 0
      || m_IntrinsicLeftFileName.size() == 0
      || m_DistortionLeftFileName.size() == 0
      || m_RotationLeftFileName.size() == 0
      || m_TranslationLeftFileName.size() == 0
      || m_IntrinsicRightFileName.size() == 0
      || m_DistortionRightFileName.size() == 0
      || m_RightToLeftRotationFileName.size() == 0
      || m_RightToLeftTranslationFileName.size() == 0
      )
  {
    throw std::logic_error("Flie names not initialised!");
  }

  m_PolyDataReader = vtkPolyDataReader::New();
  m_PolyDataReader->SetFileName(m_Input3DModelFileName.c_str());
  m_PolyDataReader->Update();

  vtkPoints *points = m_PolyDataReader->GetOutput()->GetPoints();
  vtkPointData *pointData = m_PolyDataReader->GetOutput()->GetPointData();

  if(points == NULL)
  {
    throw std::logic_error("Model does not have points! Object is NULL.");
  }

  if (points->GetNumberOfPoints() == 0)
  {
    throw std::logic_error("Model does not have any points! Object contains zero points");
  }

  if(pointData == NULL)
  {
    throw std::logic_error("Model does not have vtkPointData!");
  }

  vtkDataArray *normals = pointData->GetNormals();

  if(points == NULL)
  {
    throw std::logic_error("Model does not have normals!");
  }

  m_InputLeftImage = cvLoadImage(m_InputLeftImageFileName.c_str());
  if (m_InputLeftImage == NULL)
  {
    throw std::logic_error("Could not load input left image!");
  }

  m_InputRightImage = cvLoadImage(m_InputRightImageFileName.c_str());
  if (m_InputRightImage == NULL)
  {
    throw std::logic_error("Could not load input right image!");
  }

  m_IntrinsicLeft = (CvMat*)cvLoad(m_IntrinsicLeftFileName.c_str());
  if (m_IntrinsicLeft == NULL)
  {
    throw std::logic_error("Failed to load left camera intrinsic params");
  }

  m_DistortionLeft = (CvMat*)cvLoad(m_DistortionLeftFileName.c_str());
  if (m_DistortionLeft == NULL)
  {
    throw std::logic_error("Failed to load left camera distortion params");
  }

  m_RotationLeft = (CvMat*)cvLoad(m_RotationLeftFileName.c_str());
  if (m_RotationLeft == NULL)
  {
    throw std::logic_error("Failed to load left camera rotation params");
  }

  m_TranslationLeft = (CvMat*)cvLoad(m_TranslationLeftFileName.c_str());
  if (m_TranslationLeft == NULL)
  {
    throw std::logic_error("Failed to load left camera translation params");
  }

  m_IntrinsicRight = (CvMat*)cvLoad(m_IntrinsicRightFileName.c_str());
  if (m_IntrinsicRight == NULL)
  {
    throw std::logic_error("Failed to load right camera intrinsic params");
  }

  m_DistortionRight = (CvMat*)cvLoad(m_DistortionRightFileName.c_str());
  if (m_DistortionRight == NULL)
  {
    throw std::logic_error("Failed to load right camera distortion params");
  }

  m_RightToLeftRotation = (CvMat*)cvLoad(m_RightToLeftRotationFileName.c_str());
  if (m_RightToLeftRotation == NULL)
  {
    throw std::logic_error("Failed to load right to left rotation params");
  }

  m_RightToLeftTranslation = (CvMat*)cvLoad(m_RightToLeftTranslationFileName.c_str());
  if (m_RightToLeftTranslation == NULL)
  {
    throw std::logic_error("Failed to load right to left translation params");
  }

  // Produce output so we can visualise it.
  m_OutputLeftImage = cvCloneImage(m_InputLeftImage);
  m_OutputRightImage = cvCloneImage(m_InputRightImage);

  // Convert VTK model to OpenCV model
  double normal[3];

  int numberOfModelPoints = points->GetNumberOfPoints();

  m_ModelPoints = cvCreateMat(numberOfModelPoints, 3, CV_32FC1);
  m_ModelNormals = cvCreateMat(numberOfModelPoints, 3, CV_32FC1);

  for (int i = 0; i < numberOfModelPoints; i++)
  {
    normals->GetTuple(i, normal);

    for (int j = 0; j < 3; j++)
    {
      CV_MAT_ELEM(*m_ModelPoints, float, i, j) = points->GetPoint(i)[j];
      CV_MAT_ELEM(*m_ModelNormals, float, i, j) = normal[j];
    }
  }

  // Create camera normal pointing along z axis.
  m_CameraNormal = cvCreateMat(1, 3, CV_32FC1);
  CV_MAT_ELEM(*m_CameraNormal, float, 0, 0) = 0;
  CV_MAT_ELEM(*m_CameraNormal, float, 0, 1) = 0;
  CV_MAT_ELEM(*m_CameraNormal, float, 0, 2) = 1;

}


//-----------------------------------------------------------------------------
void StereoImageToModelMetric::PrintSelf(std::ostream& os, itk::Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "m_Input3DModel: " << m_Input3DModelFileName << std::endl;
  os << indent << "m_InputLeftImageName: " << m_InputLeftImageFileName << std::endl;
  os << indent << "m_InputRightImageName: " << m_InputRightImageFileName << std::endl;
  os << indent << "m_OutputLeftImageName: " << m_OutputLeftImageFileName << std::endl;
  os << indent << "m_OutputRightImageName: " << m_OutputRightImageFileName << std::endl;
  os << indent << "m_IntrinsicLeftFileName: " << m_IntrinsicLeftFileName << std::endl;
  os << indent << "m_DistortionLeftFileName: " << m_DistortionLeftFileName << std::endl;
  os << indent << "m_RotationLeftFileName: " << m_RotationLeftFileName << std::endl;
  os << indent << "m_TranslationLeftFileName: " << m_TranslationLeftFileName << std::endl;
  os << indent << "m_IntrinsicRightFileName: " << m_IntrinsicRightFileName << std::endl;
  os << indent << "m_DistortionRightFileName: " << m_DistortionRightFileName << std::endl;
  os << indent << "m_RightToLeftRotationFileName: " << m_RightToLeftRotationFileName << std::endl;
  os << indent << "m_RightToLeftTranslationFileName: " << m_RightToLeftTranslationFileName << std::endl;
}


//-----------------------------------------------------------------------------
void StereoImageToModelMetric::GetDerivative( const StereoImageToModelMetric::ParametersType & parameters,
    StereoImageToModelMetric::DerivativeType & derivative ) const
{
  for (unsigned int i = 0; i < parameters.GetSize(); i++)
  {
    StereoImageToModelMetric::ParametersType offset = parameters;
    offset[i] += 1;

    double plusValue = this->GetValue(offset);

    offset[i] -= 2;

    double minusValue = this->GetValue(offset);

    derivative[i] = (plusValue-minusValue / 2.0);
  }
}


//-----------------------------------------------------------------------------
StereoImageToModelMetric::MeasureType StereoImageToModelMetric::GetValue( const ParametersType &parameters ) const
{
  StereoImageToModelMetric::MeasureType currentValue = 0;

  CvMat *transformationMatrix = Construct4x4TransformationMatrixFromDegrees(
      parameters[0], // rx
      parameters[1], // ry
      parameters[2], // rz
      parameters[3], // tx
      parameters[4], // ty
      parameters[5]  // tz
      );

  int numberOfModelPoints = m_PolyDataReader->GetOutput()->GetPoints()->GetNumberOfPoints();

  // Transform points into camera view.
  CvMat *transformedPoints = cvCreateMat(numberOfModelPoints, 3, CV_32FC1);
  CvMat *transformedNormals = cvCreateMat(numberOfModelPoints, 3, CV_32FC1);

  TransformBy4x4Matrix(*m_ModelPoints, *transformationMatrix, false, *transformedPoints);
  TransformBy4x4Matrix(*m_ModelNormals, *transformationMatrix, true, *transformedNormals);

  // Create these pointers ... but not the matrices, as ProjectVisible3DWorldPointsToStereo2D will do memory allocation.
  CvMat *outputLeftCameraWorldPointsIn3D = NULL;
  CvMat *outputLeftCameraWorldNormalsIn3D = NULL;
  CvMat *output2DPointsLeft = NULL;
  CvMat *output2DPointsRight = NULL;

  int numberProjectedPoints = 0;
  std::vector<int> validPointIds;

  // Project registered model onto output image, which allocates the output arrays.
  validPointIds = ProjectVisible3DWorldPointsToStereo2D(
      *transformedPoints,
      *transformedNormals,
      *m_CameraNormal,
      *m_IntrinsicLeft,
      *m_DistortionLeft,
      *m_IntrinsicRight,
      *m_DistortionRight,
      *m_RightToLeftRotation,
      *m_RightToLeftTranslation,
      outputLeftCameraWorldPointsIn3D,
      outputLeftCameraWorldNormalsIn3D,
      output2DPointsLeft,
      output2DPointsRight
      );

  numberProjectedPoints = validPointIds.size();

  if (numberProjectedPoints > 0)
  {

    // Sanity check
    if (outputLeftCameraWorldPointsIn3D->rows != numberProjectedPoints
        || outputLeftCameraWorldNormalsIn3D->rows != numberProjectedPoints
        || output2DPointsLeft->rows != numberProjectedPoints
        || output2DPointsRight->rows != numberProjectedPoints
        )
    {
      throw std::logic_error("Invalid number of points in 2D/3D");
    }

    vtkDataArray *weights = m_PolyDataReader->GetOutput()->GetPointData()->GetScalars();

    CvMat *outputPointWeights = cvCreateMat(numberProjectedPoints, 1, CV_32FC1);
    for (int i = 0; i < numberProjectedPoints; i++)
    {
      CV_MAT_ELEM(*outputPointWeights, float, i, 0) = (float)(*(weights->GetTuple(i)));
    }

    // Call derived class for similarity measure.
    currentValue = CalculateCost(
        *outputLeftCameraWorldPointsIn3D,
        *outputLeftCameraWorldNormalsIn3D,
        *outputPointWeights,
        *output2DPointsLeft,
        *output2DPointsRight,
        parameters
        );

    if (m_DrawOutput)
    {
      // Draw circle for each projected point
      for (int i = 0; i < numberProjectedPoints; i++)
      {
        cvCircle(m_OutputLeftImage, cvPoint(CV_MAT_ELEM(*output2DPointsLeft, float, i, 0), CV_MAT_ELEM(*output2DPointsLeft, float, i, 1)), 1, CV_RGB(255,0,0), 1, 8);
        cvCircle(m_OutputRightImage, cvPoint(CV_MAT_ELEM(*output2DPointsRight, float, i, 0), CV_MAT_ELEM(*output2DPointsRight, float, i, 1)), 1, CV_RGB(255,0,0), 1, 8);
      }

      // Save output images.
      cvSaveImage(m_OutputLeftImageFileName.c_str(), m_OutputLeftImage);
      cvSaveImage(m_OutputRightImageFileName.c_str(), m_OutputRightImage);
    }

    // Store params, which are mutable.
    m_Parameters = parameters;

    // Tidy up.
    cvReleaseMat(&transformationMatrix);
    cvReleaseMat(&transformedPoints);
    cvReleaseMat(&transformedNormals);
    cvReleaseMat(&outputLeftCameraWorldPointsIn3D);
    cvReleaseMat(&outputLeftCameraWorldNormalsIn3D);
    cvReleaseMat(&outputPointWeights);
    cvReleaseMat(&output2DPointsLeft);
    cvReleaseMat(&output2DPointsRight);

  }
  else
  {
    throw std::logic_error("Failed to project any points");
  }

  if(this->GetDebug())
  {
    std::cout << "Cost:" << currentValue << ", from:" << parameters[0] \
        << ", " << parameters[1] \
        << ", " << parameters[2] \
        << ", " << parameters[3] \
        << ", " << parameters[4] \
        << ", " << parameters[5] \
        << std::endl;
  }

  return currentValue;
}


//-----------------------------------------------------------------------------
bool StereoImageToModelMetric::GetImageValues(const float &lx, const float &ly, const float &rx, const float &ry, float *leftValue, float *rightValue) const
{
  bool successful = true;
  successful &= GetImageValue(const_cast<IplImage*>(this->m_InputLeftImage), lx, ly, leftValue);
  successful &= GetImageValue(const_cast<IplImage*>(this->m_InputRightImage), rx, ry, rightValue);
  return successful;
}


//-----------------------------------------------------------------------------
bool StereoImageToModelMetric::GetImageValue(const IplImage* image, const float &x, const float &y, float *imageValue) const
{
  bool successful = true;

  int xint = (int)x;
  int yint = (int)y;
  int nChannels = image->nChannels;

  if (xint < 0 || yint < 0 || xint >= image->width-1 || yint >= image->width-1)
  {
    for (int i = 0; i < nChannels; i++)
    {
      imageValue[i] = 0;
      successful = false;
    }
  }
  else
  {
    float w1 = x - (float)xint;
    float w2 = (float)(xint + 1) - x;
    float w3 = y - (float)(yint);
    float w4 = (float)(yint + 1) - y;

    for (int i = 0; i < nChannels; i++)
    {
      imageValue[i] = image->imageData[yint*image->widthStep + xint + i]*w2*w4
                    + image->imageData[yint*image->widthStep + xint+1 + i]*w1*w4
                    + image->imageData[(yint+1)*image->widthStep + xint + i]*w2*w3
                    + image->imageData[(yint+1)*image->widthStep + (xint+1) + i]*w1*w3;
    }
  }
  return successful;
}

//-----------------------------------------------------------------------------
} // end namespace mitk


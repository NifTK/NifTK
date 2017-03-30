/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include "niftkUltrasoundProcessing.h"
#include <Internal/niftkQuaternion.h>
#include <mitkExceptionMacro.h>
#include <mitkIOUtil.h>
#include <mitkConvert2Dto3DImageFilter.h>
#include <niftkOpenCVImageConversion.h>
#include <niftkMITKMathsUtils.h>
#include <niftkFileHelper.h>
#include <niftkFileIOUtils.h>
#include <mitkOpenCVMaths.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>
#include <mitkImageToItk.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageRegionIterator.h>
#include <vtkMath.h>
#include <itkCastImageFilter.h>


namespace niftk
{

typedef std::pair<niftkQuaternion, niftkQuaternion> TrackingQuaternions;

typedef unsigned char InputPixelType;
typedef double OutputPixelType;

const unsigned int dim = 3;

typedef itk::Image<InputPixelType, dim> InputImageType;
typedef itk::Image<OutputPixelType, dim> OutputImageType;
typedef itk::Image<unsigned char, dim> ResultImageType;


//-------------------------------------------------------------------------------------
int HoughForRadius(const cv::Mat& image, int x, int y, int& max_radius, int medianR)
{
  int i;
  int j;
  int hist[10];
  int pixel_num[10];
  int radius;
  int max_weight = 0;

  for (i = 0; i < 10; i++)
  {
    hist[i] = pixel_num[i] = 0;
  }

  max_radius = 0;
  int outerR = medianR + 5;
  int innerR = medianR - 5;

  for (j = y - outerR; j <= y + outerR; j++)
  {
    for (i = x - outerR; i <= x + outerR; i++)
    {
      if (j < 0 || j >= int(image.rows) || i < 0 || i >= int(image.cols))
      {
        continue;
      }

      radius = int(sqrt((j - y) * (j - y) + (i - x) * (i - x)) + 0.5);

      if ((radius >= innerR) && (radius < outerR))
      {
        hist[radius - innerR] += *(image.data + j * image.cols + i);
        pixel_num[radius - innerR]++;
      }
    }
  }

  for (i = 0; i < 10; i++)
  {
    int value = hist[i] / pixel_num[i];

    if (value > max_weight)
    {
      max_weight = value;
      max_radius = i + innerR;
    }
  }

  return max_weight;
}


//------------------------------------------------------------------------------
void RawHough(const cv::Mat& image, int& x, int& y, int& r, int medianR)
{
  int max_x = x;
  int max_y = y;
  int max_radius = 0;
  int radius = 0;
  int weight;
  int max_weight = 0;

  for (int j = y - 5; j <= y + 5; j++)
  {
    for (int i = x - 5; i <= x + 5; i++)
    {
      weight = HoughForRadius(image, i, j, radius, medianR);

      if (weight > max_weight)
      {
        max_weight = weight;
        max_radius = radius;
        max_x = i;
        max_y = j;
      }
    }
  }

  x = max_x;
  y = max_y;
  r = max_radius;
  
  return;
}


//------------------------------------------------------------------------------
cv::Mat CreateRingModel(const int model_width)
{
  cv::Mat model(model_width, model_width, CV_8U, cv::Scalar(0));

  int innerR = (model_width - 30) / 2;
  int outerR = model_width / 2;

  int outerR2 = outerR * outerR;
  int innerR2 = innerR * innerR;

  cv::MatIterator_<unsigned char> iter = model.begin<unsigned char>();

  for (int i = 0; i < model_width; i++)
  {
    for (int j = 0; j < model_width; j++, iter++)
    {
      int d2 = (i - outerR) * (i - outerR) + (j - outerR) * (j - outerR);

      if ((d2 <= outerR2) && (d2 >= innerR2))
      {
        *iter = 255;
      }
    }
  }

  return model;
}


//-----------------------------------------------------------------------------
mitk::Point2D FindCircleInImage(const cv::Mat& image, const cv::Mat& model)
{
  int image_width = image.cols;
  int image_height = image.rows;
  int model_width = model.cols;

  // Consider out-of-picture balls, allowing up to 1/4 of the ball going out of the side of the picture
  int startx = - model_width / 4;
  int starty = 0; // The top of the ball would always be seen
  int endx = image_width - 1 - model_width * 3 / 4;
  int endy = image_height - 1 - model_width * 3 / 4;

  double min_diff = std::numeric_limits<double>::max();
  int max_x = 0;
  int max_y = 0;
  int max_radius = 0;

  unsigned char *ptr_image;
  unsigned char *ptr_model;

  // Multiresolution template matching, using down-sampling
  for (int rate = 4; rate >= 1; rate /= 2)
  {
    for (int j = starty; j <= endy; j += rate)
    {
      for (int i = startx; i <= endx; i += rate)
      {
        ptr_image = image.data + j * image_width;
        ptr_model = model.data;

        double  val2 = 0;
        int  pixel_num = 0;

        for (int n = 0; (n < model_width) && ( j + n < image_height);
          n += rate, ptr_image += image_width * rate, ptr_model += model_width * rate)
        {
          for (int m = i > 0 ? 0 : -i; (m < model_width) && ( i + m < image_width); m += rate)
          {
            if (*(ptr_model + m) > 0)
            {
              double val = *(ptr_image + m+i) - *(ptr_model + m);
              val2 += val * val;
              pixel_num++;
            }
          }
        }

        val2 /= pixel_num;

        if (val2 < min_diff)
        {
          min_diff = val2;
          max_x = i;
          max_y = j;
        }
      }// end for i
    }// end for j

    startx = max_x - rate / 2;
    starty = max_y - rate / 2;

    endx = max_x + rate / 2;
    endy = max_y + rate / 2;

    min_diff = std::numeric_limits<double>::max();
  }// end for rate

  max_x += (model_width - 1) / 2;
  max_y += (model_width - 1) / 2;

  int medianR;
  medianR = (model_width - 15) / 2;

  RawHough(image, max_x, max_y, max_radius, medianR);

  mitk::Point2D result;
  result[0] = max_x;
  result[1] = max_y;

  return result;
}


//---------------------------------------------------------------------------
cv::Mat UltrasoundCalibration(const TrackedPointData& trackedPoints)
{
  int number_of_scans = (int)trackedPoints.size();

  cv::Mat F(3 * number_of_scans, 1, CV_64F);
  cv::Mat Ftmp(3 * number_of_scans, 1, CV_64F);
  cv::Mat J(3 * number_of_scans, 12, CV_64F);
  cv::Mat H(12, 12, CV_64F);

  niftkQuaternion q1;
  niftkQuaternion t1;
  niftkQuaternion t3;
  niftkQuaternion qx;

  // Parameter initialisation
  cv::Mat a(12, 1, CV_64F, cv::Scalar(0));

  double sx = a.at<double>(0) = 1.0;
  double sy = a.at<double>(1) = 1.0;

  q1[0] = a.at<double>(2) = 1.0;
  q1[1] = a.at<double>(3) = 0.0;
  q1[2] = a.at<double>(4) = 0.0;
  q1[3] = a.at<double>(5) = 0.0;

  t1[0] = 0.0;
  t1[1] = a.at<double>(6) = 0.0;
  t1[2] = a.at<double>(7) = 0.0;
  t1[3] = a.at<double>(8) = 0.0;

  t3[0] = 0.0;
  t3[1] = a.at<double>(9) = 0.0;
  t3[2] = a.at<double>(10) = 0.0;
  t3[3] = a.at<double>(11) = 0.0;

  niftkQuaternion fi;

  vector<niftkQuaternion> Q2;
  vector<niftkQuaternion> T2;
  
  for(auto iter = trackedPoints.begin(); iter != trackedPoints.end(); iter++)
  {
    niftkQuaternion r;
    r[0] = iter->second.first[0];
    r[1] = iter->second.first[1];
    r[2] = iter->second.first[2];
    r[3] = iter->second.first[3];

    niftkQuaternion t;
    t[0] = 0;
    t[1] = iter->second.second[0];
    t[2] = iter->second.second[1];
    t[3] = iter->second.second[2];

    Q2.push_back(r);
    T2.push_back(t);
  }

  double p22;
  double p23;
  double p24;

  double p32;
  double p33;
  double p34;

  double p42;
  double p43;
  double p44;

  double dqxq_ds_21;
  double dqxq_ds_22;

  double dqxq_ds_31;
  double dqxq_ds_32;

  double dqxq_ds_41;
  double dqxq_ds_42;

  double dqxq_dq_21;
  double dqxq_dq_22;
  double dqxq_dq_23;
  double dqxq_dq_24;

  double dqxq_dq_31;
  double dqxq_dq_32;
  double dqxq_dq_33;
  double dqxq_dq_34;

  double dqxq_dq_41;
  double dqxq_dq_42;
  double dqxq_dq_43;
  double dqxq_dq_44;

  double lambda = 0.001;
  cv::Mat I = cv::Mat::eye(12, 12, CV_64FC1);

  for (int times = 0; times < 100; times++) // 100 iteratios for the LM algorithm
  {
    for (int i = 0; i < number_of_scans; i++)
    {
      qx[0] = 0;
      qx[1] = sx * trackedPoints[i].first[0];
      qx[2] = sy * trackedPoints[i].first[1];
      qx[3] = 0;

      p22 = Q2[i][0] * Q2[i][0]
      + Q2[i][1] * Q2[i][1]
      - Q2[i][2] * Q2[i][2]
      - Q2[i][3] * Q2[i][3];
      
      p23 = 2 * Q2[i][1] * Q2[i][2]
      - 2 * Q2[i][0] * Q2[i][3];
      
      p24 = 2 * Q2[i][1] * Q2[i][3]
      + 2 * Q2[i][0] * Q2[i][2];

      p32 = 2 * Q2[i][1] * Q2[i][2]
      + 2 * Q2[i][0] * Q2[i][3];

      p33 = Q2[i][0] * Q2[i][0]
      + Q2[i][2] * Q2[i][2]
      - Q2[i][1] * Q2[i][1]
      - Q2[i][3] * Q2[i][3];

      p34 = 2 * Q2[i][2] * Q2[i][3]
      - 2 * Q2[i][0] * Q2[i][1];

      p42 = 2 * Q2[i][1] * Q2[i][3]
      - 2 * Q2[i][0] * Q2[i][2];

      p43 = 2 * Q2[i][2] * Q2[i][3]
      + 2 * Q2[i][0] * Q2[i][1];
      
      p44 = Q2[i][0] * Q2[i][0]
      + Q2[i][3] * Q2[i][3]
      - Q2[i][1] * Q2[i][1]
      - Q2[i][2] * Q2[i][2];

      dqxq_ds_21 = q1[1] * q1[1] * trackedPoints[i].first[0]
        + q1[0] * q1[0] * trackedPoints[i].first[0]
        - q1[2] * q1[2] * trackedPoints[i].first[0]
        - q1[3] * q1[3] * trackedPoints[i].first[0];

      dqxq_ds_22 = 2 * q1[1] * q1[2] * trackedPoints[i].first[1]
        - 2 * q1[0] * q1[3] * trackedPoints[i].first[1];

      dqxq_ds_31 = 2 * q1[1] * q1[2] * trackedPoints[i].first[0]
        + 2 * q1[0] * q1[3] * trackedPoints[i].first[0];

      dqxq_ds_32 = q1[2] * q1[2] * trackedPoints[i].first[1]
        + q1[0] * q1[0] * trackedPoints[i].first[1]
        - q1[1] * q1[1] * trackedPoints[i].first[1]
        - q1[3] * q1[3] * trackedPoints[i].first[1];

      dqxq_ds_41 = 2 * q1[1] * q1[3] * trackedPoints[i].first[0]
        - 2 * q1[0] * q1[2] * trackedPoints[i].first[0];

      dqxq_ds_42 = 2 * q1[2] * q1[3] * trackedPoints[i].first[1]
        + 2 * q1[0] * q1[1] * trackedPoints[i].first[1];

      dqxq_dq_21 = 2 * q1[0] * sx * trackedPoints[i].first[0] - 2 * q1[3] * sy * trackedPoints[i].first[1];
      dqxq_dq_22 = 2 * q1[1] * sx * trackedPoints[i].first[0] + 2 * q1[2] * sy * trackedPoints[i].first[1];
      dqxq_dq_23 = -2 * q1[2] * sx * trackedPoints[i].first[0] + 2 * q1[1] * sy * trackedPoints[i].first[1];
      dqxq_dq_24 = -2 * q1[3] * sx * trackedPoints[i].first[0] - 2 * q1[0] * sy * trackedPoints[i].first[1];

      dqxq_dq_31 = 2 * q1[3] * sx * trackedPoints[i].first[0] + 2 * q1[0] * sy * trackedPoints[i].first[1];
      dqxq_dq_32 = 2 * q1[2] * sx * trackedPoints[i].first[0] - 2 * q1[1] * sy * trackedPoints[i].first[1];
      dqxq_dq_33 = 2 * q1[1] * sx * trackedPoints[i].first[0] + 2 * q1[2] * sy * trackedPoints[i].first[1];
      dqxq_dq_34 = 2 * q1[0] * sx * trackedPoints[i].first[0] - 2 * q1[3] * sy * trackedPoints[i].first[1];

      dqxq_dq_41 = -2 * q1[2] * sx * trackedPoints[i].first[0] + 2 * q1[1] * sy * trackedPoints[i].first[1];
      dqxq_dq_42 = 2 * q1[3] * sx * trackedPoints[i].first[0] + 2 * q1[0] * sy * trackedPoints[i].first[1];
      dqxq_dq_43 = -2 * q1[0] * sx * trackedPoints[i].first[0] + 2 * q1[3] * sy * trackedPoints[i].first[1];
      dqxq_dq_44 = 2 * q1[1] * sx * trackedPoints[i].first[0] + 2 * q1[2] * sy * trackedPoints[i].first[1];

      J.at<double>(3 * i, 0) = p22 * dqxq_ds_21 + p23 * dqxq_ds_31 + p24 * dqxq_ds_41;
      J.at<double>(3 * i, 1) = p22 * dqxq_ds_22 + p23 * dqxq_ds_32 + p24 * dqxq_ds_42;

      J.at<double>(3 * i, 2) = p22 * dqxq_dq_21 + p23 * dqxq_dq_31 + p24 * dqxq_dq_41;
      J.at<double>(3 * i, 3) = p22 * dqxq_dq_22 + p23 * dqxq_dq_32 + p24 * dqxq_dq_42;
      J.at<double>(3 * i, 4) = p22 * dqxq_dq_23 + p23 * dqxq_dq_33 + p24 * dqxq_dq_43;
      J.at<double>(3 * i, 5) = p22 * dqxq_dq_24 + p23 * dqxq_dq_34 + p24 * dqxq_dq_44;

      J.at<double>(3 * i, 6) = p22;
      J.at<double>(3 * i, 7) = p23;
      J.at<double>(3 * i, 8) = p24;

      J.at<double>(3 * i, 9) = 1;
      J.at<double>(3 * i, 10) = 0;
      J.at<double>(3 * i, 11) = 0;

      J.at<double>(3 * i + 1, 0) = p32 * dqxq_ds_21 + p33 * dqxq_ds_31 + p34 * dqxq_ds_41;
      J.at<double>(3 * i + 1, 1) = p32 * dqxq_ds_22 + p33 * dqxq_ds_32 + p34 * dqxq_ds_42;

      J.at<double>(3 * i + 1, 2) = p32 * dqxq_dq_21 + p33 * dqxq_dq_31 + p34 * dqxq_dq_41;
      J.at<double>(3 * i + 1, 3) = p32 * dqxq_dq_22 + p33 * dqxq_dq_32 + p34 * dqxq_dq_42;
      J.at<double>(3 * i + 1, 4) = p32 * dqxq_dq_23 + p33 * dqxq_dq_33 + p34 * dqxq_dq_43;
      J.at<double>(3 * i + 1, 5) = p32 * dqxq_dq_24 + p33 * dqxq_dq_34 + p34 * dqxq_dq_44;

      J.at<double>(3 * i + 1, 6) = p32;
      J.at<double>(3 * i + 1, 7) = p33;
      J.at<double>(3 * i + 1, 8) = p34;

      J.at<double>(3 * i + 1, 9) = 0;
      J.at<double>(3 * i + 1, 10) = 1;
      J.at<double>(3 * i + 1, 11) = 0;

      J.at<double>(3 * i + 2, 0) = p42 * dqxq_ds_21 + p43 * dqxq_ds_31 + p44 * dqxq_ds_41;
      J.at<double>(3 * i + 2, 1) = p42 * dqxq_ds_22 + p43 * dqxq_ds_32 + p44 * dqxq_ds_42;

      J.at<double>(3 * i + 2, 2) = p42 * dqxq_dq_21 + p43 * dqxq_dq_31 + p44 * dqxq_dq_41;
      J.at<double>(3 * i + 2, 3) = p42 * dqxq_dq_22 + p43 * dqxq_dq_32 + p44 * dqxq_dq_42;
      J.at<double>(3 * i + 2, 4) = p42 * dqxq_dq_23 + p43 * dqxq_dq_33 + p44 * dqxq_dq_43;
      J.at<double>(3 * i + 2, 5) = p42 * dqxq_dq_24 + p43 * dqxq_dq_34 + p44 * dqxq_dq_44;

      J.at<double>(3 * i + 2, 6) = p42;
      J.at<double>(3 * i + 2, 7) = p43;
      J.at<double>(3 * i + 2, 8) = p44;

      J.at<double>(3 * i + 2, 9) = 0;
      J.at<double>(3 * i + 2, 10) = 0;
      J.at<double>(3 * i + 2, 11) = 1;

      fi = Q2[i] * (q1 * qx * q1.Conjugate() + t1) * Q2[i].Conjugate() + T2[i] + t3;

      F.at<double>(i * 3) = fi[1];
      F.at<double>(i * 3 + 1) = fi[2];
      F.at<double>(i * 3 + 2) = fi[3];
    }// end for i

    if (times != 0)
    {
      if (norm(F) > norm(Ftmp))
      {
        lambda *= 10;
      }
      else
      {
        lambda /= 10;
      }
    }

    Ftmp = F;

/*********************With Colume Normalisation of J*******************************
    H = ColumnScaler(J);
    Mat JH = J * H; //Normalise each column of J
    Mat JHT = JH.t();

    a += H * (JHT * JH + I * lambda).inv(DECOMP_SVD) * JHT * (-F);
**********************************************************************************/

/*********************Without Colume Normalisation of J*******************************/
    cv::Mat JT = J.t();

    a += (JT * J + I * lambda).inv(cv::DECOMP_SVD) * JT * (-F);
/**********************************************************************************/

    // Modifying
    a.at<double>(0) = fabs(a.at<double>(0));
    a.at<double>(1) = fabs(a.at<double>(1));

    // Normalisng into unit quaternion
    double q_norm = sqrt(a.at<double>(2) * a.at<double>(2) + a.at<double>(3) * a.at<double>(3)
      + a.at<double>(4) * a.at<double>(4) + a.at<double>(5) * a.at<double>(5));

    q1[0] = a.at<double>(2) / q_norm;
    q1[1] = a.at<double>(3) / q_norm;
    q1[2] = a.at<double>(4) / q_norm;
    q1[3] = a.at<double>(5) / q_norm;

    a.at<double>(2) = q1[0];
    a.at<double>(3) = q1[1];
    a.at<double>(4) = q1[2];
    a.at<double>(5) = q1[3];

    sx = a.at<double>(0);
    sy = a.at<double>(1);

    t1[0] = 0.0;
    t1[1] = a.at<double>(6);
    t1[2] = a.at<double>(7);
    t1[3] = a.at<double>(8);

    t3[0] = 0.0;
    t3[1] = a.at<double>(9);
    t3[2] = a.at<double>(10);
    t3[3] = a.at<double>(11);
  }//end for times

  double MSE =  norm(F) / sqrt(3 * number_of_scans);

  cout << "Mean Square Eoor: " << MSE << std::endl;
  return a;
}


/*
The diameter of the circle in the images should be measured with an interactive tool
Ring model width = diameter + 15
*/
void DoUltrasoundBallCalibration(const int& ballSize,
                                 const niftk::TrackedImageData& trackedImages,
                                 mitk::Point2D& pixelScaleFactors,
                                 RotationTranslation& imageToSensorTransform
                                )
{

  MITK_INFO << "DoUltrasoundBallCalibration: Doing Ultrasound Ball Calibration with "
            << trackedImages.size() << " samples.";

  TrackedPointData trackedPoints;

  cv::Mat model = CreateRingModel(ballSize + 15);

  // Extract all 2D centres of circles
  for (int i = 0; i < trackedImages.size(); i++)
  {
    cv::Mat tmpImage = niftk::MitkImageToOpenCVMat(trackedImages[i].first);
    mitk::Point2D pixelLocation = niftk::FindCircleInImage(tmpImage, model);

    cout << "Circle " << i << " found" << std::endl;

    TrackedPoint aTrackedPoint;

    aTrackedPoint.first = pixelLocation;
    aTrackedPoint.second = trackedImages[i].second;

    trackedPoints.push_back(aTrackedPoint);
  }

  // Now do calibration.
  cv::Mat parameters = UltrasoundCalibration(trackedPoints);

  // Now copy into output
  pixelScaleFactors[0] = parameters.at<double>(0);
  pixelScaleFactors[1] = parameters.at<double>(1);

  imageToSensorTransform.first[0] = parameters.at<double>(2);
  imageToSensorTransform.first[1] = parameters.at<double>(3);
  imageToSensorTransform.first[2] = parameters.at<double>(4);
  imageToSensorTransform.first[3] = parameters.at<double>(5);

  imageToSensorTransform.second[0] = parameters.at<double>(6);
  imageToSensorTransform.second[1] = parameters.at<double>(7);
  imageToSensorTransform.second[2] = parameters.at<double>(8);
}


//-------------------------------------------------------------------------------------------------------
void DoUltrasoundPointCalibration(const niftk::TrackedPointData& trackedPoints,
                                  mitk::Point2D& pixelScaleFactors,
                                  RotationTranslation& imageToSensorTransform
                                  )
{
  MITK_INFO << "DoUltrasoundPointCalibration: Doing Ultrasound Point Calibration with "
            << trackedPoints.size() << " samples.";

  cv::Mat parameters = UltrasoundCalibration(trackedPoints);

  // Now copy into output
  pixelScaleFactors[0] = parameters.at<double>(0);
  pixelScaleFactors[1] = parameters.at<double>(1);

  imageToSensorTransform.first[0] = parameters.at<double>(2);
  imageToSensorTransform.first[1] = parameters.at<double>(3);
  imageToSensorTransform.first[2] = parameters.at<double>(4);
  imageToSensorTransform.first[3] = parameters.at<double>(5);

  imageToSensorTransform.second[0] = parameters.at<double>(6);
  imageToSensorTransform.second[1] = parameters.at<double>(7);
  imageToSensorTransform.second[2] = parameters.at<double>(8);
}


/******************************Reconstruction***********************************/
void DoUltrasoundReconstructionFor1Slice(InputImageType::Pointer itk2D,
                                         OutputImageType::Pointer accumulator,
                                         OutputImageType::Pointer itk3D
                                         )
{
  typedef itk::ImageRegionConstIteratorWithIndex<InputImageType> ConstIteratorType;

  ConstIteratorType inputIter(itk2D, itk2D->GetRequestedRegion());

  for (inputIter.GoToBegin(); !inputIter.IsAtEnd(); ++inputIter)
  {
    InputPixelType pixelValue = inputIter.Get();
    if (pixelValue == 0)
    {
      continue; // Ignore black areas
    }

    InputImageType::IndexType idx2D = inputIter.GetIndex();
    OutputImageType::IndexType idx3D;

    OutputImageType::PointType pt;

    itk2D->TransformIndexToPhysicalPoint(idx2D, pt);
    itk3D->TransformPhysicalPointToIndex(pt, idx3D);

    OutputPixelType voxelValue = itk3D->GetPixel(idx3D);

    OutputPixelType currentWeight = accumulator->GetPixel(idx3D);
    currentWeight = currentWeight + 1;

    itk3D->SetPixel(idx3D, voxelValue + pixelValue);
    accumulator->SetPixel(idx3D, currentWeight);
  }

}


//-----------------------------------------------------------------------------
mitk::Image::Pointer DoUltrasoundReconstruction(const niftk::TrackedImageData& data,
                                                const mitk::Point2D& pixelScaleFactors,
                                                const niftk::RotationTranslation& imageToSensorTransform,
                                                const mitk::Vector3D& voxelSpacing
                                                )
{
  MITK_INFO << "DoUltrasoundReconstruction: Doing Ultrasound Reconstruction with "
            << data.size() << " samples.";

  if (data.size() == 0)
  {
    mitkThrow() << "No reconstruction data provided.";
  }

  mitk::PixelType unsignedCharPixelType = mitk::MakeScalarPixelType<unsigned char>();

  if (data[0].first->GetPixelType() != unsignedCharPixelType)
  {
    mitkThrow() << "Ultrasound images should be unsigned char.";
  }

  if (data[0].first->GetPixelType().GetNumberOfComponents() != 1)
  {
    mitkThrow() << "Ultrasound images should have 1 component (i.e. greyscale not RGB)";
  }

  vtkSmartPointer<vtkMatrix4x4> scalingMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  scalingMatrix->Identity();
  scalingMatrix->SetElement(0, 0, pixelScaleFactors[0]);
  scalingMatrix->SetElement(1, 1, pixelScaleFactors[1]);

  vtkSmartPointer<vtkMatrix4x4> imageToSensorMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  niftk::ConvertRotationAndTranslationToMatrix(imageToSensorTransform.first,
                                               imageToSensorTransform.second,
                                               *imageToSensorMatrix
                                               );

  vtkSmartPointer<vtkMatrix4x4> pixelToSensorMatrix = vtkSmartPointer<vtkMatrix4x4>::New();

  vtkMatrix4x4::Multiply4x4(imageToSensorMatrix, scalingMatrix, pixelToSensorMatrix);

  vtkSmartPointer<vtkMatrix4x4> trackingMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  trackingMatrix->Identity();

  vtkSmartPointer<vtkMatrix4x4> indexToWorld = vtkSmartPointer<vtkMatrix4x4>::New();
  indexToWorld->Identity();

  // Calculate size of bounding box, in millimetres.
  // Here we use scaling, imageToSensor transform and tracking (sensorToWorld) transform.
  mitk::Point3D minCornerInMillimetres;
  minCornerInMillimetres[0] = std::numeric_limits<double>::max();
  minCornerInMillimetres[1] = std::numeric_limits<double>::max();
  minCornerInMillimetres[2] = std::numeric_limits<double>::max();

  mitk::Point3D maxCornerInMillimetres;
  maxCornerInMillimetres[0] = -1 * std::numeric_limits<double>::max();
  maxCornerInMillimetres[1] = -1 * std::numeric_limits<double>::max();
  maxCornerInMillimetres[2] = -1 * std::numeric_limits<double>::max();

  mitk::Point3I cornersIndexArray[4];
  mitk::Point3D cornersWorldArray[4];
 
  for (unsigned int num = 0; num < data.size(); num++)
  {
    if (data[num].first.IsNull())
    {
      mitkThrow() << "Ultrasound image " << num << " is NULL?!?!?";
    }
    if (data[num].first->GetDimension() != 3)
    {
      mitkThrow() << "Ultrasound images should be 3D.";
    }
    if (data[num].first->GetDimensions()[2] != 1)
    {
      mitkThrow() << "Ultrasound images should be 3D, with 1 slice.";
    }

    niftk::ConvertRotationAndTranslationToMatrix(data[num].second.first,
                                                 data[num].second.second,
                                                 *trackingMatrix);

    vtkMatrix4x4::Multiply4x4(trackingMatrix, pixelToSensorMatrix, indexToWorld);

    unsigned int *dims = data[num].first->GetDimensions();
    assert(dims[2] == 1); // only checks in Debug mode though.

    cornersIndexArray[0][0] = 0;
    cornersIndexArray[0][1] = 0;
    cornersIndexArray[0][2] = 0;

    cornersIndexArray[1][0] = dims[0];
    cornersIndexArray[1][1] = 0;
    cornersIndexArray[1][2] = 0;

    cornersIndexArray[2][0] = 0;
    cornersIndexArray[2][1] = dims[1];
    cornersIndexArray[2][2] = 0;

    cornersIndexArray[3][0] = dims[0];
    cornersIndexArray[3][1] = dims[1];
    cornersIndexArray[3][2] = 0;

    mitk::BaseGeometry* imageGeometry = data[num].first->GetGeometry();
    imageGeometry->SetIndexToWorldTransformByVtkMatrix(indexToWorld);

    // Multiply min, max pixel index by indexToWorld
    // Check for most negative and most positive x,y,z coordinate.
    // Store in minCornerInMillimetres, maxCornerInMillimetres.
    for (int i = 0; i < 4; i++)
    {
      imageGeometry->IndexToWorld(cornersIndexArray[i], cornersWorldArray[i]);

      for (int j = 0; j < 3; j++)
      {
        if (cornersWorldArray[i][j] < minCornerInMillimetres[j])
        {
          minCornerInMillimetres[j] = cornersWorldArray[i][j];
        }
        
        if (cornersWorldArray[i][j] > maxCornerInMillimetres[j])
        {
          maxCornerInMillimetres[j] = cornersWorldArray[i][j];
        }
      }// end for j
    }// end for i
  } // end for num

  mitk::Point3D origin;
  origin[0] = minCornerInMillimetres[0] - (0.5 * voxelSpacing[0]); // Origin position in millimetres.
  origin[1] = minCornerInMillimetres[1] - (0.5 * voxelSpacing[1]); // Origin position in millimetres.
  origin[2] = minCornerInMillimetres[2] - (0.5 * voxelSpacing[2]); // Origin position in millimetres.

  unsigned int dims[3];
  dims[0] = (maxCornerInMillimetres[0] - minCornerInMillimetres[0]) / voxelSpacing[0] + 2; // Number of voxels in x
  dims[1] = (maxCornerInMillimetres[1] - minCornerInMillimetres[1]) / voxelSpacing[1] + 2; // Number of voxels in y
  dims[2] = (maxCornerInMillimetres[2] - minCornerInMillimetres[2]) / voxelSpacing[2] + 2; // Number of voxels in z

  MITK_INFO << "DoUltrasoundReconstruction creating 2 volumes of ("
            << dims[0] << ", " << dims[1] << ", " << dims[2] << "), "
            << "with resolution "
            << voxelSpacing[0] << "x" << voxelSpacing[1] << "x" << voxelSpacing[2]
            << "mm." << std::endl;

  // Working images are double.
  mitk::PixelType doublePixelType = mitk::MakeScalarPixelType<double>();

  // This will be the output image.
  mitk::Image::Pointer image3D = mitk::Image::New();
  image3D->Initialize(doublePixelType, 3, dims);
  image3D->SetSpacing(voxelSpacing);
  image3D->SetOrigin(origin);

  mitk::Image::Pointer accumulatorImage = mitk::Image::New();
  accumulatorImage->Initialize(doublePixelType, 3, dims);
  accumulatorImage->SetSpacing(voxelSpacing);
  accumulatorImage->SetOrigin(origin);

  mitk::Vector3D pixelSpacing;
  pixelSpacing[0] = pixelScaleFactors[0]; // Size of 2D pixels in x in millimetres
  pixelSpacing[1] = pixelScaleFactors[1]; // Size of 2D pixels in y in millimetres
  pixelSpacing[2] = 1.0;                  // Size of 2D pixels in z in millimetres
                                          // - any value will do as the z coordinate is zero

  auto itk3D = mitk::ImageToItkImage< OutputPixelType, dim >(image3D);
  auto accumulator = mitk::ImageToItkImage< OutputPixelType, dim >(accumulatorImage);

  OutputImageType::DirectionType itk3DImageDirection;
  itk3DImageDirection.SetIdentity();

  itk3D->SetDirection(itk3DImageDirection);
  accumulator->SetDirection(itk3DImageDirection);

  itk3D->FillBuffer(0.0);
  accumulator->FillBuffer(0.0);

  double quaternion[4];

  quaternion[0] = imageToSensorTransform.first[0];
  quaternion[1] = imageToSensorTransform.first[1];
  quaternion[2] = imageToSensorTransform.first[2];
  quaternion[3] = imageToSensorTransform.first[3];

  double rotationMatrix[3][3];
  vtkMath::QuaternionToMatrix3x3(quaternion, rotationMatrix);
  cv::Mat calibratedRotation(3, 3, CV_64F, rotationMatrix);

  cv::Mat calibratedTranslation(3, 1, CV_64F);
  for (int r = 0; r < 3; r++)
  {
    calibratedTranslation.at<double>(r) = imageToSensorTransform.second[r];
  }

  // Now iterate through each image/tracking data pair, and put in volume.
  for (int i = 0; i < data.size(); i++)
  {
    double quaternion[4];
    quaternion[0] = data[i].second.first[0];
    quaternion[1] = data[i].second.first[1];
    quaternion[2] = data[i].second.first[2];
    quaternion[3] = data[i].second.first[3];

    double rotationMatrix[3][3];
    vtkMath::QuaternionToMatrix3x3(quaternion, rotationMatrix);
    cv::Mat trackingRotation(3, 3, CV_64F, rotationMatrix);

    cv::Mat trackingTranslation(3, 1, CV_64F);
    for (int r = 0; r < 3; r++)
    {
      trackingTranslation.at<double>(r) = data[i].second.second[r];
    }

    cv::Mat newOrigin(3, 1, CV_64F);
    cv::Mat newDirection(3, 3, CV_64F);

    newOrigin = trackingTranslation + trackingRotation * calibratedTranslation;
    newDirection = trackingRotation * calibratedRotation;

    mitk::Image::Pointer image2D = data[i].first;
    image2D->SetSpacing(pixelSpacing);

    mitk::Point3D newOrigin2D;
    newOrigin2D[0] = newOrigin.at<double>(0);
    newOrigin2D[1] = newOrigin.at<double>(1);
    newOrigin2D[2] = newOrigin.at<double>(2);

    image2D->SetOrigin(newOrigin2D);

    InputImageType::Pointer itk2D = mitk::ImageToItkImage< InputPixelType, dim >(image2D);

    InputImageType::DirectionType itk2DImageDirection;
    itk2DImageDirection = itk2D->GetDirection();

    for (int row = 0; row < 3; row++)
    {
      for (int col = 0; col < 3; col++)
      {
        itk2DImageDirection[row][col] = newDirection.at<double>(row, col);
      }
    }

    itk2D->SetDirection(itk2DImageDirection);

    DoUltrasoundReconstructionFor1Slice(itk2D,
                                        accumulator,
                                        itk3D);

    cout << "Slice " << i << " reconstructed" << std::endl;

  }// end for i

  //Do averaging
  typedef itk::ImageRegionIterator<OutputImageType> OutputIteratorType;

  OutputIteratorType itk3DIter(itk3D, itk3D->GetRequestedRegion());
  OutputIteratorType accumulatorIter(accumulator, itk3D->GetRequestedRegion());

  for (itk3DIter.GoToBegin(), accumulatorIter.GoToBegin(); !itk3DIter.IsAtEnd(); ++itk3DIter, ++accumulatorIter)
  {
    OutputPixelType totalValue = itk3DIter.Get();
    OutputPixelType totalWeight = accumulatorIter.Get();

    if (totalValue > TINY_NUMBER && totalWeight > TINY_NUMBER)
    {
      itk3DIter.Set(totalValue / totalWeight);
    }
  }

  //Cast voxel type from double to unsigned char
  typedef itk::CastImageFilter< OutputImageType, ResultImageType > FilterType;
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(itk3D);
  filter->Update();

  mitk::Image::Pointer resultImage = mitk::Image::New();
  auto filteredImage = filter->GetOutput();
  resultImage->InitializeByItk(filteredImage);
  resultImage->SetVolume(filteredImage->GetBufferPointer());
  resultImage->SetGeometry(image3D->GetGeometry());

  // And returns the image.
  return resultImage;
}


//-----------------------------------------------------------------------------------------------
void LoadOxfordQuaternionTrackingFile(std::string filename,
                                      mitk::Point4D& rotation,
                                      mitk::Vector3D& translation
                                      )
{
  fstream fp(filename, ios::in);
  
  if (!fp)
  {
    std::ostringstream errorMessage;
    errorMessage << "Can't open quaternion tracking data file " << filename << std::endl;
    mitkThrow() << errorMessage.str();
  }

  char line[130];
  fp.getline(line, 130, '\n');
  fp.getline(line, 130, '\n');
  fp.getline(line, 130, '\n');

  fp >> translation[0] >> translation[1] >> translation[2] >> rotation[0] >> rotation[1] >> rotation[2] >> rotation[3];

  fp.close();

  return;
}


/* For each time-stamped file in fileList1, find the closest time-matched file in fileList2*/
//---------------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string>> PairTimeStampedDataFiles(const std::vector<std::string>& fileList1,
                                                                          const std::vector<std::string>& fileList2
                                                                          )
{
  std::vector<std::pair<std::string, std::string>> pairedFiles;

  int matchNumber = 0;

  // For debugging
  fstream fp("pairs.txt", ios::out);
  if (!fp)
  {
    std::ostringstream errorMessage;
    errorMessage << "Can't write pairing result!" << std::endl;
    mitkThrow() << errorMessage.str();
  }

  for (int i = 0; i < fileList1.size(); i++)
  {
    std::size_t found = fileList1[i].find_last_of(".");

    // Time stamp used
//    std::string firstFileTimeStamp = fileList1[i].substr(found - 30, 12); // For folder UltrasonixRemote_2
    std::string firstFileTimeStamp = fileList1[i].substr(found - 14, 12); // For normal time-stamped names

    long long int minTimeDifference = std::numeric_limits<long long int>::max();
    long long int firstFileTime = std::stoll(firstFileTimeStamp);

    std::string matchTime; // For debugging

    long long int closestTime = 0;

    for (int j = matchNumber; j < fileList2.size(); j++)
    {
      found = fileList2[j].find_last_of(".");

      std::string secondFileTimeStamp = fileList2[j].substr(found - 14, 12);

      long long int secondFileTime = std::stoll(secondFileTimeStamp);
      long long int timeDifference = abs(secondFileTime - firstFileTime);

      if (timeDifference == 0) // Time matched exactly!
      {
        matchNumber = j;
        matchTime = fileList2[matchNumber].substr(found - 14, 12); // For debugging
        closestTime  = timeDifference;
        break;
      }

      if (timeDifference < minTimeDifference)
      {
        minTimeDifference = timeDifference;

        if (j != fileList2.size() - 1)
        {
          continue;
        }
        else
        {
          matchNumber = j;
          matchTime = fileList2[matchNumber].substr(found - 14, 12); // For debugging
          closestTime  = minTimeDifference;
          break;
        }
      }
      else
      {
        matchNumber = j - 1;
        matchTime = fileList2[matchNumber].substr(found - 14, 12); // For debugging
        closestTime  = minTimeDifference;
        break;
      }
    } // end for j

    if (closestTime > 1000000) // This file has no closely matched tracking file, discarded
    {
      continue;
    }

    std::pair<std::string, std::string> aPairOfFiles;
    aPairOfFiles.first = fileList1[i];
    aPairOfFiles.second = fileList2[matchNumber];

    pairedFiles.push_back(aPairOfFiles);

    // For debugging
    cout << "File:" << "\t" << i << "\t" << firstFileTime << std::endl;
    cout << "Match:" << "\t" << matchNumber << "\t" << matchTime << std::endl;
    cout << "Lag:" << "\t" << closestTime << std::endl;

    fp << "File:" << "\t" << i << "\t" << firstFileTime << std::endl;
    fp << "Match:" << "\t" << matchNumber << "\t" << matchTime << std::endl;
    fp << "Lag:" << "\t" << closestTime << std::endl;
  } // end for i

  fp.close();

  return pairedFiles;
}


// Pair and load time-stamped points and tracking data from directories.
// Not applicable to old Oxford data
TrackedPointData LoadPointAndTrackingDataFromDirectories(const std::string& pointDir,
                                                         const std::string& trackingDir
                                                         )
{
  std::vector<std::string> pointFiles = niftk::GetFilesInDirectory(pointDir);
  std::vector<std::string> trackingFiles = niftk::GetFilesInDirectory(trackingDir);
  
  std::size_t found = pointFiles[0].find_last_of(".");
  std::string ext = pointFiles[0].substr(found + 1);

  if (( ext != "txt") && ( ext != "4x4"))
  {
    std::ostringstream errorMessage;
    errorMessage << pointFiles[0] << " is not a point file. Wrong directory?" << std::endl;
    mitkThrow() << errorMessage.str();
  }

  found = trackingFiles[0].find_last_of(".");
  ext = trackingFiles[0].substr(found + 1);

  if (( ext != "txt") && ( ext != "4x4"))
  {
    std::ostringstream errorMessage;
    errorMessage << trackingFiles[0] << " is not a tracking file. Wrong directory?" << std::endl;
    mitkThrow() << errorMessage.str();
  }

  if (pointFiles.size() > trackingFiles.size())
  {
    std::ostringstream errorMessage;
    errorMessage << "The number of point files should not be more than the number of tracking files." << std::endl;
    mitkThrow() << errorMessage.str();
  }
  
  std::vector<std::pair<std::string, std::string>> pairedFiles;

  if (pointFiles.size() == trackingFiles.size())
  {
    for (int i = 0; i < pointFiles.size(); i++)
    {
       std::pair<std::string, std::string> aPairOfFiles;

       aPairOfFiles.first = pointFiles[i];
       aPairOfFiles.second = trackingFiles[i];

       pairedFiles.push_back(aPairOfFiles);
    }
  }
  else
  {
    pairedFiles = PairTimeStampedDataFiles(pointFiles, trackingFiles);
  }

  TrackedPointData outputData;

  for (int i = 0; i < pairedFiles.size(); i++)
  {
    // Read the point file -- Need to be changed to be able to read 2D points as well
    mitk::Point3D aPoint3D;
    if(!niftk::Load3DPointFromFile(pairedFiles[i].first, aPoint3D))
    {
      std::ostringstream errorMessage;
      errorMessage << "Can not read point file " << pairedFiles[i].first << std::endl;
      mitkThrow() << errorMessage.str();
    }

    mitk::Point2D aPoint2D;
    aPoint2D[0] = aPoint3D[0];
    aPoint2D[1] = aPoint3D[1];

    // Read the matched tracking data. If in matrix format, convert to quaternions
    mitk::Point4D rotation;
    mitk::Vector3D translation;

    found = pairedFiles[i].second.find_last_of(".");
    ext = pairedFiles[i].second.substr(found + 1);

    if (( ext == "txt") || ( ext == "4x4"))
    {
      vtkSmartPointer<vtkMatrix4x4> trackingMatrix = niftk::LoadVtkMatrix4x4FromFile(pairedFiles[i].second);

      //Convert to quaternions
      niftk::ConvertMatrixToRotationAndTranslation(*trackingMatrix, rotation, translation);
    }
    else
    {
      std::ostringstream errorMessage;
      errorMessage << "Unknown tracking data type in " << pairedFiles[i].second << std::endl;
      mitkThrow() << errorMessage.str();
    }

    RotationTranslation aRotationTranslationPair;
    aRotationTranslationPair.first = rotation;
    aRotationTranslationPair.second = translation;

    TrackedPoint aTrackedPoint;
    aTrackedPoint.first = aPoint2D;
    aTrackedPoint.second = aRotationTranslationPair;

    outputData.push_back(aTrackedPoint);
  } // end for i

  if (outputData.size() < 30)
  {
    std::ostringstream errorMessage;
    errorMessage << "Not enough matched points and tracking data for calibration!" << std::endl;
    mitkThrow() << errorMessage.str();
  }

  return outputData;
}


//-----------------------------------------------------------------------------
TrackedImageData LoadImageAndTrackingDataFromDirectories(const std::string& imageDir,
                                                         const std::string& trackingDir
                                                         )
{
  std::vector<std::string> imageFiles = niftk::GetFilesInDirectory(imageDir);
  std::vector<std::string> trackingFiles = niftk::GetFilesInDirectory(trackingDir);

  std::size_t found = imageFiles[0].find_last_of(".");
  std::string  ext = imageFiles[0].substr(found + 1);

  if (( ext != "png") && ( ext != "nii")) // Need to include more image formats...
  {
    std::ostringstream errorMessage;
    errorMessage << imageFiles[0] << " is not an image file. Wrong directory?" << std::endl;
    mitkThrow() << errorMessage.str();
  }

  found = trackingFiles[0].find_last_of(".");
  ext = trackingFiles[0].substr(found + 1);

  if (( ext != "txt") && ( ext != "4x4") && ( ext != "pos"))
  {
    std::ostringstream errorMessage;
    errorMessage << trackingFiles[0] << " is not a tracking file. Wrong directory?" << std::endl;
    mitkThrow() << errorMessage.str();
  }

  if (imageFiles.size() > trackingFiles.size())
  {
    std::ostringstream errorMessage;
    errorMessage << "The number of images should not be more than the number of tracking data." << std::endl;
    mitkThrow() << errorMessage.str();
  }

  std::vector<std::pair<std::string, std::string>> pairedFiles;

  if ((ext == "pos") || imageFiles.size() == trackingFiles.size()) // Oxford data are already paired
  {
    for (int i = 0; i < imageFiles.size(); i++)
    {
       std::pair<std::string, std::string> aPairOfFiles;

       aPairOfFiles.first = imageFiles[i];
       aPairOfFiles.second = trackingFiles[i];

       pairedFiles.push_back(aPairOfFiles);
    }
  }
  else
  {
    pairedFiles = PairTimeStampedDataFiles(imageFiles, trackingFiles);
  }

  TrackedImageData outputData;

  // Load all images using mitk::IOUtil, assuming there is enough memory
  // Load tracking data and if of matrix type, convert to quaternions
  for (int i = 0; i < pairedFiles.size(); i++)
  {
    // Load one image file
    mitk::Image::Pointer tmpImage = mitk::IOUtil::LoadImage(pairedFiles[i].first);
    mitk::Convert2Dto3DImageFilter::Pointer filter = mitk::Convert2Dto3DImageFilter::New();
    filter->SetInput(tmpImage);
    filter->Update();

    mitk::Image::Pointer convertedImage = filter->GetOutput();

    // Load one tracking file
    mitk::Point4D rotation;
    mitk::Vector3D translation;

    found = pairedFiles[i].second.find_last_of(".");
    ext = pairedFiles[i].second.substr(found + 1);

    if (( ext == "txt") || ( ext == "4x4"))
    {
      vtkSmartPointer<vtkMatrix4x4> trackingMatrix = niftk::LoadVtkMatrix4x4FromFile(pairedFiles[i].second);

      //Convert to quaternions
      niftk::ConvertMatrixToRotationAndTranslation(*trackingMatrix, rotation, translation);
    }
    else
      if ( ext == "pos") // For Oxford data, in quaternions
      {
        LoadOxfordQuaternionTrackingFile(pairedFiles[i].second, rotation, translation);
      }
      else
      {
        std::ostringstream errorMessage;
        errorMessage << "Unknown tracking data type in " << pairedFiles[i].second << std::endl;
        mitkThrow() << errorMessage.str();
      }

    RotationTranslation aRotationTranslationPair;
    aRotationTranslationPair.first = rotation;
    aRotationTranslationPair.second = translation;

    TrackedImage aTrackedImage;

    aTrackedImage.first = convertedImage;
    aTrackedImage.second = aRotationTranslationPair;

    outputData.push_back(aTrackedImage);
  }

  // this will incur a copy, but you wont copy images, you will copy smart pointers.
  return outputData;
}

} // end namespace

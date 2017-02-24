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


namespace niftk
{
//-------------------------------------------------------------------------------------
int HoughForRadius(const cv::Mat& image, int x, int y, int& max_radius, int medianR)
{
  int i;
  int j;
  int hist[10];
  int pixel_num[10];
  int radius;
  int max_weight = 0;

  for (i = 0; i<10; i++)
    hist[i] = pixel_num[i] = 0;

  max_radius = 0;
  int outerR = medianR + 5;
  int innerR = medianR - 5;

  for (j = y - outerR; j <= y + outerR; j++)
    for (i = x - outerR; i <= x + outerR; i++)
    {
      if (j<0 || j >= int(image.rows) || i<0 || i >= int(image.cols))
        continue;

      radius = int(sqrt((j - y)*(j - y) + (i - x)*(i - x)) + 0.5);

      if ((radius >= innerR) && (radius<outerR))
      {
        hist[radius - innerR] += *(image.data + j*image.cols + i);
        pixel_num[radius - innerR]++;
      }
    }

  for (i = 0; i<10; i++)
  {
    int value = hist[i] / pixel_num[i];

    if (value>max_weight)
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
    for (int i = x - 5; i <= x + 5; i++)
    {
      weight = HoughForRadius(image, i, j, radius, medianR);

      if (weight>max_weight)
      {
        max_weight = weight;
        max_radius = radius;
        max_x = i;
        max_y = j;
      }
    }

  x = max_x;
  y = max_y;
  r = max_radius;
  
  return;
}


//Create a ring model for template matching
cv::Mat CreateRingModel(const int model_width)
{
  cv::Mat model(model_width, model_width, CV_8U, cv::Scalar(0));

  int innerR = (model_width - 30) / 2;
  int outerR = model_width / 2;

  int outerR2 = outerR * outerR;
  int innerR2 = innerR * innerR;

  cv::MatIterator_<unsigned char> iter = model.begin<unsigned char>();

  for (int i = 0; i<model_width; i++)
    for (int j = 0; j<model_width; j++, iter++)
    {
      int d2 = (i - outerR)*(i - outerR) + (j - outerR)*(j - outerR);

      if ((d2 <= outerR2) && (d2 >= innerR2))
        *iter = 255;
    }

  return model;
}


//-----------------------------------------------------------------------------
cv::Point2d FindCircleInImage(const cv::Mat& image, cv::Mat& model)
{
  int image_width = image.cols;
  int image_height = image.rows;
  int model_width = model.cols;

  //Consider out-of-picture balls, allowing up to 1/4 of the ball going out of the side of the picture
  int startx = - model_width / 4;
  int starty = 0; //The top of the ball would always be seen
  int endx = image_width - 1 - model_width * 3 / 4;
  int endy = image_height - 1 - model_width * 3 / 4;

  double min_diff = 1000000000;
  int max_x = 0;
  int max_y = 0;
  int max_radius = 0;

  unsigned char *ptr_image;
  unsigned char *ptr_model;

  //Multiresolution template matching, using down-sampling
  for (int rate = 4; rate >= 1; rate /= 2)
  {
    for (int j = starty; j <= endy; j += rate)
    {
      for (int i = startx; i <= endx; i += rate)
      {
        ptr_image = image.data + j*image_width;
        ptr_model = model.data;

        double  val2 = 0;
        int  pixel_num = 0;

        for (int n = 0; (n<model_width) && ( j+n < image_height);
          n += rate, ptr_image += image_width*rate, ptr_model += model_width*rate)
          for (int m = i>0 ? 0 : -i; (m<model_width) && ( i+m < image_width); m += rate)
          {
            if (*(ptr_model+m) > 0)
            {
              double val = *(ptr_image + m+i) - *(ptr_model + m);
              val2 += val * val;
              pixel_num++;
            }
          }

        val2 /= pixel_num;

        if (val2 < min_diff)
        {
          min_diff = val2;
          max_x = i;
          max_y = j;
        }
      }//end for i,j
    }

    startx = max_x - rate / 2;
    starty = max_y - rate / 2;

    endx = max_x + rate / 2;
    endy = max_y + rate / 2;

    min_diff = 1000000000;
  }// end for rate

  max_x += (model_width - 1) / 2;
  max_y += (model_width - 1) / 2;

  int medianR;
  medianR = (model_width - 15) / 2;

  RawHough(image, max_x, max_y, max_radius, medianR);

  cv::Point2d result;
  result.x = max_x;
  result.y = max_y;

  return result;
}


// tracking_data is a pair of quaternions representing rotation and translation
// Caution: the old .para files record translation first and then a unit quaternion for rotation  
cv::Mat UltrasoundCalibration(const std::vector<cv::Point2d>& points,
                              const std::vector<TrackingQuaternions>& tracking_data)
{
  int number_of_scans = (int)points.size();

  cv::Mat F(3 * number_of_scans, 1, CV_64F);
  cv::Mat Ftmp(3 * number_of_scans, 1, CV_64F);
  cv::Mat J(3 * number_of_scans, 12, CV_64F);
  cv::Mat H(12, 12, CV_64F);

  niftkQuaternion q1;
  niftkQuaternion t1;
  niftkQuaternion t3;
  niftkQuaternion qx;

  //Parameter initialisation
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
  
  for(auto iter = tracking_data.begin(); iter != tracking_data.end(); iter++)
  {
    Q2.push_back(iter->first);
    T2.push_back(iter->second);
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

  for (int times = 0; times<100; times++) //100 iteratios for the LM algorithm
  {
    for (int i = 0; i<number_of_scans; i++)
    {
      qx[0] = 0;
      qx[1] = sx*points[i].x;
      qx[2] = sy*points[i].y;
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

      dqxq_ds_21 = q1[1] * q1[1] * points[i].x
        + q1[0] * q1[0] * points[i].x
        - q1[2] * q1[2] * points[i].x
        - q1[3] * q1[3] * points[i].x;

      dqxq_ds_22 = 2 * q1[1] * q1[2] * points[i].y
        - 2 * q1[0] * q1[3] * points[i].y;

      dqxq_ds_31 = 2 * q1[1] * q1[2] * points[i].x
        + 2 * q1[0] * q1[3] * points[i].x;

      dqxq_ds_32 = q1[2] * q1[2] * points[i].y
        + q1[0] * q1[0] * points[i].y
        - q1[1] * q1[1] * points[i].y
        - q1[3] * q1[3] * points[i].y;

      dqxq_ds_41 = 2 * q1[1] * q1[3] * points[i].x
        - 2 * q1[0] * q1[2] * points[i].x;

      dqxq_ds_42 = 2 * q1[2] * q1[3] * points[i].y
        + 2 * q1[0] * q1[1] * points[i].y;

      dqxq_dq_21 = 2 * q1[0] * sx*points[i].x - 2 * q1[3] * sy*points[i].y;
      dqxq_dq_22 = 2 * q1[1] * sx*points[i].x + 2 * q1[2] * sy*points[i].y;
      dqxq_dq_23 = -2 * q1[2] * sx*points[i].x + 2 * q1[1] * sy*points[i].y;
      dqxq_dq_24 = -2 * q1[3] * sx*points[i].x - 2 * q1[0] * sy*points[i].y;

      dqxq_dq_31 = 2 * q1[3] * sx*points[i].x + 2 * q1[0] * sy*points[i].y;
      dqxq_dq_32 = 2 * q1[2] * sx*points[i].x - 2 * q1[1] * sy*points[i].y;
      dqxq_dq_33 = 2 * q1[1] * sx*points[i].x + 2 * q1[2] * sy*points[i].y;
      dqxq_dq_34 = 2 * q1[0] * sx*points[i].x - 2 * q1[3] * sy*points[i].y;

      dqxq_dq_41 = -2 * q1[2] * sx*points[i].x + 2 * q1[1] * sy*points[i].y;
      dqxq_dq_42 = 2 * q1[3] * sx*points[i].x + 2 * q1[0] * sy*points[i].y;
      dqxq_dq_43 = -2 * q1[0] * sx*points[i].x + 2 * q1[3] * sy*points[i].y;
      dqxq_dq_44 = 2 * q1[1] * sx*points[i].x + 2 * q1[2] * sy*points[i].y;

      J.at<double>(3 * i, 0) = p22*dqxq_ds_21 + p23*dqxq_ds_31 + p24*dqxq_ds_41;
      J.at<double>(3 * i, 1) = p22*dqxq_ds_22 + p23*dqxq_ds_32 + p24*dqxq_ds_42;

      J.at<double>(3 * i, 2) = p22*dqxq_dq_21 + p23*dqxq_dq_31 + p24*dqxq_dq_41;
      J.at<double>(3 * i, 3) = p22*dqxq_dq_22 + p23*dqxq_dq_32 + p24*dqxq_dq_42;
      J.at<double>(3 * i, 4) = p22*dqxq_dq_23 + p23*dqxq_dq_33 + p24*dqxq_dq_43;
      J.at<double>(3 * i, 5) = p22*dqxq_dq_24 + p23*dqxq_dq_34 + p24*dqxq_dq_44;

      J.at<double>(3 * i, 6) = p22;
      J.at<double>(3 * i, 7) = p23;
      J.at<double>(3 * i, 8) = p24;

      J.at<double>(3 * i, 9) = 1;
      J.at<double>(3 * i, 10) = 0;
      J.at<double>(3 * i, 11) = 0;

      J.at<double>(3 * i + 1, 0) = p32*dqxq_ds_21 + p33*dqxq_ds_31 + p34*dqxq_ds_41;
      J.at<double>(3 * i + 1, 1) = p32*dqxq_ds_22 + p33*dqxq_ds_32 + p34*dqxq_ds_42;

      J.at<double>(3 * i + 1, 2) = p32*dqxq_dq_21 + p33*dqxq_dq_31 + p34*dqxq_dq_41;
      J.at<double>(3 * i + 1, 3) = p32*dqxq_dq_22 + p33*dqxq_dq_32 + p34*dqxq_dq_42;
      J.at<double>(3 * i + 1, 4) = p32*dqxq_dq_23 + p33*dqxq_dq_33 + p34*dqxq_dq_43;
      J.at<double>(3 * i + 1, 5) = p32*dqxq_dq_24 + p33*dqxq_dq_34 + p34*dqxq_dq_44;

      J.at<double>(3 * i + 1, 6) = p32;
      J.at<double>(3 * i + 1, 7) = p33;
      J.at<double>(3 * i + 1, 8) = p34;

      J.at<double>(3 * i + 1, 9) = 0;
      J.at<double>(3 * i + 1, 10) = 1;
      J.at<double>(3 * i + 1, 11) = 0;

      J.at<double>(3 * i + 2, 0) = p42*dqxq_ds_21 + p43*dqxq_ds_31 + p44*dqxq_ds_41;
      J.at<double>(3 * i + 2, 1) = p42*dqxq_ds_22 + p43*dqxq_ds_32 + p44*dqxq_ds_42;

      J.at<double>(3 * i + 2, 2) = p42*dqxq_dq_21 + p43*dqxq_dq_31 + p44*dqxq_dq_41;
      J.at<double>(3 * i + 2, 3) = p42*dqxq_dq_22 + p43*dqxq_dq_32 + p44*dqxq_dq_42;
      J.at<double>(3 * i + 2, 4) = p42*dqxq_dq_23 + p43*dqxq_dq_33 + p44*dqxq_dq_43;
      J.at<double>(3 * i + 2, 5) = p42*dqxq_dq_24 + p43*dqxq_dq_34 + p44*dqxq_dq_44;

      J.at<double>(3 * i + 2, 6) = p42;
      J.at<double>(3 * i + 2, 7) = p43;
      J.at<double>(3 * i + 2, 8) = p44;

      J.at<double>(3 * i + 2, 9) = 0;
      J.at<double>(3 * i + 2, 10) = 0;
      J.at<double>(3 * i + 2, 11) = 1;

      fi = Q2[i] * (q1*qx*q1.Conjugate() + t1)*Q2[i].Conjugate() + T2[i] + t3;

      F.at<double>(i * 3) = fi[1];
      F.at<double>(i * 3 + 1) = fi[2];
      F.at<double>(i * 3 + 2) = fi[3];
    }// end for i

    if (times != 0)
    {
      if ( norm(F) > norm(Ftmp) )
        lambda *= 10;
      else
        lambda /= 10;
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

    //Modifying
    a.at<double>(0) = fabs(a.at<double>(0));
    a.at<double>(1) = fabs(a.at<double>(1));

    //Normalisng into unit quaternion
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

  double LSE =  norm(F) / sqrt(3 * number_of_scans);

  return a;
}


//Using quaternion representaion-------------------------------------------------------------------
void DoUltrasoundCalibration(const QuaternionTrackedImageData& data,
                             double& pixelToMillimetreScaleX,
                             double& pixelToMillimetreScaleY,
                             TrackingQuaternions& imageToSensorTransform
                             )
{
  MITK_INFO << "DoUltrasoundCalibration: Doing Ultrasound Calibration with "
            << data.size() << " samples.";

  std::vector<cv::Point2d> points;
  std::vector<TrackingQuaternions> tracking_data;

  int model_width = 300; // Input the roughly measured circle diameter
  cv::Mat model = CreateRingModel(model_width);

  // Extract all 2D centres of circles
  for (int i = 0; i < data.size(); i++)
  {
    cv::Mat tmpImage = niftk::MitkImageToOpenCVMat(data[i].first);
    cv::Point2d pixelLocation = niftk::FindCircleInImage(tmpImage, model);

    points.push_back(pixelLocation);
    tracking_data.push_back(data[i].second);
  }

  // Now do calibration.
  cv::Mat parameters = UltrasoundCalibration(points, tracking_data);

  // Now copy into output
  pixelToMillimetreScaleX = parameters.at<double>(0);
  pixelToMillimetreScaleY = parameters.at<double>(1);

  imageToSensorTransform.first[0] = parameters.at<double>(2);
  imageToSensorTransform.first[1] = parameters.at<double>(3);
  imageToSensorTransform.first[2] = parameters.at<double>(4);
  imageToSensorTransform.first[3] = parameters.at<double>(5);

  imageToSensorTransform.second[0] = 0;
  imageToSensorTransform.second[1] = parameters.at<double>(6);
  imageToSensorTransform.second[2] = parameters.at<double>(7);
  imageToSensorTransform.second[3] = parameters.at<double>(8);

}

//-----------------------------------------------------------------------------
std::vector<double> UltrasoundCalibration(const std::vector<cv::Point2d>& points,
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

  int model_width = 300; // Input the roughly measured circle diameter
  cv::Mat model = CreateRingModel(model_width);

  // Extract all 2D centres of circles
  for (int i = 0; i < data.size(); i++)
  {
    // Feel free to use other OpenCV types.
    // These are just some of the examples in Niftk.

    cv::Mat tmpImage = niftk::MitkImageToOpenCVMat(data[i].first);
    cv::Point2d pixelLocation = niftk::FindCircleInImage(tmpImage, model);

    vtkSmartPointer<vtkMatrix4x4> vtkMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
    data[i].second->GetVtkMatrix(*vtkMatrix);

    cv::Matx44d trackingMatrix;
    mitk::CopyToOpenCVMatrix(*vtkMatrix, trackingMatrix);

    points.push_back(pixelLocation);
    matrices.push_back(trackingMatrix);
  }

  // Now do calibration.
  // Feel free to change return types.
  std::vector<double> parameters = UltrasoundCalibration(points, matrices);

  // Now copy into output VTK matrices
  pixelToMillimetreScale.Identity();
  pixelToMillimetreScale.SetElement(0, 0, 1 /* set scale here */);
  pixelToMillimetreScale.SetElement(1, 1, 1 /* set scale here */);

  imageToSensorTransform.Identity();
  /* set rigid body matrix here */
}


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


} // end namespace

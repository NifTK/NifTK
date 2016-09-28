/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include "niftkCaffeFCNSegmentor.h"
#include <mitkExceptionMacro.h>
#include <mitkImageWriteAccessor.h>
#include <mitkIOUtil.h>
#include <niftkOpenCVImageConversion.h>
#include <highgui.h>

namespace niftk
{

//-----------------------------------------------------------------------------
CaffeFCNSegmentor::CaffeFCNSegmentor(const std::string& networkDescriptionFileName,  // Purposefully hidden.
                                     const std::string& networkWeightsFileName,
                                     const std::string& inputLayerName,
                                     const std::string& outputBlobName
                                    )
: m_InputLayerName(inputLayerName)
, m_OutputBlobName(outputBlobName)
, m_Net(nullptr)
, m_InputBlob(nullptr)
, m_InputLabel(nullptr)
{
/*
  m_OffsetRGB[0] = -127.34116;
  m_OffsetRGB[1] = -189.179;
  m_OffsetRGB[2] = -197.013;
*/

  m_OffsetRGB[0] = -128.65884;
  m_OffsetRGB[1] = -66.821;
  m_OffsetRGB[2] = -58.987;

  m_Margin[0] = 5;
  m_Margin[1] = 5;

  caffe::Caffe::set_mode(caffe::Caffe::CPU);
  m_Net.reset(new caffe::Net<float>(networkDescriptionFileName, caffe::TEST));
  m_Net->CopyTrainedLayersFrom(networkWeightsFileName);

  if (!m_Net->has_layer(m_OutputBlobName))
  {
    mitkThrow() << "Can't find output blob:" << m_OutputBlobName
                << " in network:" << networkDescriptionFileName;
  }

  boost::shared_ptr<caffe::MemoryDataLayer<float> > memoryLayer =
      boost::dynamic_pointer_cast <caffe::MemoryDataLayer<float> >(m_Net->layer_by_name(m_InputLayerName));

  if (!memoryLayer.get())
  {
    mitkThrow() << "Can't find input layer:" << m_InputLayerName
                << " in network:" << networkDescriptionFileName;
  }

  m_OffsetValueImage = cvCreateMat(memoryLayer->height(), memoryLayer->width(), CV_32FC3);
  m_OffsetValueImage.setTo(cv::Vec3f(m_OffsetRGB[0], m_OffsetRGB[1], m_OffsetRGB[2]));
  m_InputImageWithOffset = cvCreateMat(memoryLayer->height(), memoryLayer->width(), CV_32FC3);
  m_InputBlob.reset(new caffe::Blob<float>(1, 3, memoryLayer->height(), memoryLayer->width()));
  m_InputLabel.reset(new caffe::Blob<float>(1, 1, 1, 1));
  m_InputLabel->mutable_cpu_data()[m_InputBlob->offset(0, 0, 0, 0)] = 1;
}


//-----------------------------------------------------------------------------
CaffeFCNSegmentor::~CaffeFCNSegmentor()
{
  // m_Net destroyed by std::unique_ptr.
  // m_InputBlob destroyed by std::unique_ptr.
}

//-----------------------------------------------------------------------------
void CaffeFCNSegmentor::ValidateInputs(const mitk::Image::Pointer& inputImage,
                                       const mitk::Image::Pointer& outputImage
                                      )
{
  if (inputImage.IsNull())
  {
    mitkThrow() << "Input image is NULL.";
  }
  if (outputImage.IsNull())
  {
    mitkThrow() << "Output image is NULL.";
  }
  if (inputImage->GetDimension(0) != outputImage->GetDimension(0))
  {
    mitkThrow() << "The supplied images have different x-dimension: "
                << inputImage->GetDimension(0) << " Vs "
                << outputImage->GetDimension(0);
  }
  if (inputImage->GetDimension(1) != outputImage->GetDimension(1))
  {
    mitkThrow() << "The supplied images have different y-dimension: "
                << inputImage->GetDimension(1) << " Vs "
                << outputImage->GetDimension(1);
  }
  if (outputImage->GetNumberOfChannels() != 1)
  {
    mitkThrow() << "The output image should be single channel, 8 bit, unsigned char";
  }

  mitk::PixelType pixelType = inputImage->GetPixelType();

  if (!(pixelType.GetPixelType() == itk::ImageIOBase::SCALAR
        && pixelType.GetComponentType() == itk::ImageIOBase::UCHAR)
      && !(pixelType.GetPixelType() == itk::ImageIOBase::RGB
           && pixelType.GetComponentType() == itk::ImageIOBase::UCHAR)
      && !(pixelType.GetPixelType() == itk::ImageIOBase::RGBA
           && pixelType.GetComponentType() == itk::ImageIOBase::UCHAR)
     )
  {
    mitkThrow() << "The input image should be an RGB or RGBA vector image, or greyscale unsigned char.";
  }
}


//-----------------------------------------------------------------------------
void CaffeFCNSegmentor::Segment(const mitk::Image::Pointer& inputImage,
                                const mitk::Image::Pointer& outputImage)
{

  this->ValidateInputs(inputImage, outputImage);

  // The memory layer tells us about the expected input image size.
  boost::shared_ptr<caffe::MemoryDataLayer<float> > memoryLayer =
      boost::dynamic_pointer_cast <caffe::MemoryDataLayer<float> >(m_Net->layer_by_name(m_InputLayerName));

  // Should not copy, due to OpenCV reference counting.
  cv::Mat wrappedImage = niftk::MitkImageToOpenCVMat(inputImage);

  // Should not keep re-allocating if m_ResizedInputImage the right size.
  cv::resize(wrappedImage, m_ResizedInputImage, cv::Size(memoryLayer->width() + (m_Margin[0]*2),
                                                         memoryLayer->height() + (m_Margin[1]*2)));

  // Crop: also, should just reference the input.
  cv::Rect roi(m_Margin[0], m_Margin[1], memoryLayer->width(), memoryLayer->height());
  m_CroppedInputImage = m_ResizedInputImage(roi);

  // As above, and images allocated in constructor, so no reallocating on the fly.
  cv::add(m_CroppedInputImage, m_OffsetValueImage, m_InputImageWithOffset, cv::noArray(), CV_32FC3);

  // Copy data to Caffe blob.
  for (int c = 0; c < 3; ++c)
  {
    for (int h = 0; h < m_InputImageWithOffset.rows; ++h)
    {
      for (int w = 0; w < m_InputImageWithOffset.cols; ++w)
      {
        m_InputBlob->mutable_cpu_data()[m_InputBlob->offset(0, c, h, w)]
          = m_InputImageWithOffset.at<cv::Vec3f>(h, w)[c];
      }
    }
  }
  memoryLayer->Reset(m_InputBlob->mutable_cpu_data(), m_InputLabel->mutable_cpu_data(), 1);

  // Runs Caffe classification.
  m_Net->Forward();

  // Get output blob.
  boost::shared_ptr<caffe::Blob<float> > outputBlob = m_Net->blob_by_name(m_OutputBlobName);
  if (   outputBlob->shape(0) != 1
      || outputBlob->shape(1) != 2
      )
  {
    mitkThrow() << "Unexpected output blob shape:" << outputBlob->shape_string();
  }

  // We want the output directly (float, 2 channel), so we can scale up, which interpolates.
  if (   m_DownSampledOutputImage.rows != outputBlob->shape(2)
      || m_DownSampledOutputImage.cols != outputBlob->shape(3)
      || m_DownSampledOutputImage.channels() != 2
      || m_DownSampledOutputImage.type() != CV_32FC2
      )
  {
    m_DownSampledOutputImage = cvCreateMat(outputBlob->shape(2), outputBlob->shape(3), CV_32FC2);
  }

  // Now fill the output float data.
  for (int c = 0; c < 2; c++)
  {
    for (int h = 0; h < m_DownSampledOutputImage.rows; ++h)
    {
      for (int w = 0; w < m_DownSampledOutputImage.cols; ++w)
      {
        m_DownSampledOutputImage.at<cv::Vec2f>(h, w)[c] =
          outputBlob->cpu_data()[outputBlob->offset(0, c, h, w)];
      }
    }
  }

  // Should not keep re-allocating if m_ResizedInputImage the right size.
  cv::resize(m_DownSampledOutputImage, m_UpSampledOutputImage, cv::Size(m_CroppedInputImage.cols, m_CroppedInputImage.rows));

  // Make sure the output image is the right size and type.
  if (   m_UpSampledPaddedOutputImage.rows != m_UpSampledOutputImage.rows + (m_Margin[0]*2)
      || m_UpSampledPaddedOutputImage.cols != m_UpSampledOutputImage.cols + (m_Margin[1]*2)
      || m_UpSampledPaddedOutputImage.channels() != 1
      || m_UpSampledPaddedOutputImage.type() != CV_8UC1
      )
  {
    m_UpSampledPaddedOutputImage = cvCreateMat(m_UpSampledOutputImage.rows + (m_Margin[0]*2),
                                               m_UpSampledOutputImage.cols + (m_Margin[1]*2), CV_8UC1);
  }

  // Now copy data out.
  cv::Point2i p;
  cv::Point2i q;
  cv::Vec2f   v;
  m_UpSampledPaddedOutputImage.setTo(0);
  for (int h = 0; h < m_UpSampledPaddedOutputImage.rows; ++h)
  {
    for (int w = 0; w < m_UpSampledPaddedOutputImage.cols; ++w)
    {
      p.x = w;
      p.y = h;
      q.x = w - m_Margin[0];
      q.y = h - m_Margin[1];

      if (roi.contains(p))
      {
        v = m_UpSampledOutputImage.at<cv::Vec2f>(q.y, q.x);
        if (v[1] > v[0])
        {
          m_UpSampledPaddedOutputImage.at<unsigned char>(h, w) = 255;
        }
      }
    }
  }

  if (   m_ResizedOutputImage.rows != wrappedImage.rows
      || m_ResizedOutputImage.cols != wrappedImage.cols
      || m_ResizedOutputImage.channels() != 1
      || m_ResizedOutputImage.type() != CV_8UC1
      )
  {
    m_ResizedOutputImage = cvCreateMat(wrappedImage.rows, wrappedImage.cols, CV_8UC1);
  }

  // Should not keep re-allocating if m_ResizedInputImage the right size.
  cv::resize(m_UpSampledPaddedOutputImage, m_ResizedOutputImage, cv::Size(wrappedImage.cols, wrappedImage.rows), 0, 0, CV_INTER_NN);

  // Copy to output. This relies on the fact that output is always 1 channel, 8 bit, uchar.
  mitk::ImageWriteAccessor writeAccess(outputImage);
  void* vPointer = writeAccess.GetData();
  memcpy(vPointer, m_ResizedOutputImage.data, m_ResizedOutputImage.rows * m_ResizedOutputImage.cols);
}

} // end namespace

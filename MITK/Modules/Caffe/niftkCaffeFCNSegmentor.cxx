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
CaffeFCNSegmentor::CaffeFCNSegmentor(const std::string&    networkDescriptionFileName,  // Purposefully hidden.
                                     const std::string&    networkWeightsFileName,
                                     const mitk::Vector3D& offsetRGB,
                                     const mitk::Point2I&  inputNetworkImageSize,
                                     const std::string&    outputLayerName,
                                     const mitk::Point2I&  outputNetworkImageSize
                                    )
: m_OffsetRGB(offsetRGB)
, m_InputNetworkImageSize(inputNetworkImageSize)
, m_OutputLayerName(outputLayerName)
, m_OutputNetworkImageSize(outputNetworkImageSize)
, m_Net(nullptr)
, m_InputBlob(nullptr)
, m_InputLabel(nullptr)
{
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
  m_Net.reset(new caffe::Net<float>(networkDescriptionFileName, caffe::TEST));
  m_Net->CopyTrainedLayersFrom(networkWeightsFileName);

  if (!m_Net->has_layer(outputLayerName))
  {
    mitkThrow() << "Can't find output:" << outputLayerName
                << ", in network:" << networkDescriptionFileName;
  }

  m_OffsetValueImage = cvCreateMat(m_InputNetworkImageSize[1], m_InputNetworkImageSize[0], CV_32FC3);
  m_OffsetValueImage.setTo(cv::Vec3f(m_OffsetRGB[0], m_OffsetRGB[1], m_OffsetRGB[2]));
  m_InputImageWithOffset = cvCreateMat(m_InputNetworkImageSize[1], m_InputNetworkImageSize[0], CV_32FC3);
  m_InputBlob.reset(new caffe::Blob<float>(1, 3, m_InputNetworkImageSize[1], m_InputNetworkImageSize[0]));
  m_InputLabel.reset(new caffe::Blob<float>(1, 1, 1, 1));
  m_InputLabel->mutable_cpu_data()[m_InputBlob->offset(0, 0, 0, 0)] = 1;
  m_DownSampledOutputImage = cvCreateMat(m_OutputNetworkImageSize[1], m_OutputNetworkImageSize[0], CV_8UC1);

  MITK_INFO << "CaffeFCNSegmentor initialised with("
            << networkDescriptionFileName << ", "
            << networkWeightsFileName
            << ", inputImageSize=" << m_InputNetworkImageSize
            << ", outputImageSize=" << m_OutputNetworkImageSize
            << ", offsetValueImageSize=" << m_OffsetValueImage.size()
            << ", inputImageWithOffsetSize=" << m_InputImageWithOffset.size()
            << ", downSampledOutputImageSize=" << m_DownSampledOutputImage.size()
            << ", offsetRGB=" << m_OffsetRGB;
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
}


//-----------------------------------------------------------------------------
boost::shared_ptr<caffe::MemoryDataLayer<float> > CaffeFCNSegmentor::GetAndValidateMemoryLayer()
{
  boost::shared_ptr<caffe::MemoryDataLayer<float> > memoryLayer =
      boost::dynamic_pointer_cast <caffe::MemoryDataLayer<float> >(m_Net->layer_by_name("data"));

  if (memoryLayer == nullptr)
  {
    mitkThrow() << "Failed to find memory layer.";
  }
  if (memoryLayer->width() != m_InputImageWithOffset.cols)
  {
    mitkThrow() << "Width doesn't match: OpenCV=" << m_InputImageWithOffset.cols << ", Caffe=" << memoryLayer->width() << std::endl;
  }
  if (memoryLayer->height() != m_InputImageWithOffset.rows)
  {
    mitkThrow() << "Height doesn't match: OpenCV=" << m_InputImageWithOffset.rows << ", Caffe=" << memoryLayer->height() << std::endl;
  }
  if (memoryLayer->channels() != m_InputImageWithOffset.channels())
  {
    mitkThrow() << "Number of channels doesn't match: OpenCV=" << m_InputImageWithOffset.channels() << ", Caffe=" << memoryLayer->channels() << std::endl;
  }

  return memoryLayer;
}


//-----------------------------------------------------------------------------
void CaffeFCNSegmentor::Segment(const mitk::Image::Pointer& inputImage,
                                const mitk::Image::Pointer& outputImage)
{

  this->ValidateInputs(inputImage, outputImage);

  // Should not copy, due to OpenCV reference counting.
  cv::Mat wrappedImage = niftk::MitkImageToOpenCVMat(inputImage);

  // Easier to check the number of resultant channels than check the MITK image.
  if (wrappedImage.channels() != 3 && wrappedImage.channels() != 4)
  {
    mitkThrow() << "The input image should be RGB or RGBA vector data.";
  }

  // Should not re-allocate if m_ResizedInputImage the right size.
  cv::resize(wrappedImage, m_ResizedInputImage, cv::Size(m_InputNetworkImageSize[0], m_InputNetworkImageSize[1]));

  // As above, and images allocated in constructor, so no reallocating on the fly.
  cv::add(m_ResizedInputImage, m_OffsetValueImage, m_InputImageWithOffset, cv::noArray(), CV_32FC3);

  // Copy data to Caffe blob - ToDo - look for faster alternative.
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

  boost::shared_ptr<caffe::MemoryDataLayer<float> > memoryLayer = this->GetAndValidateMemoryLayer();
  memoryLayer->Reset(m_InputBlob->mutable_cpu_data(), m_InputLabel->mutable_cpu_data(), 1);

  // Runs Caffe classification.
  m_Net->Forward();

  double total = 0;
  boost::shared_ptr<caffe::Blob<float> > conv0 = m_Net->blob_by_name("finalC");
  std::cerr << "Matt, finalC=" << conv0->shape_string() << std::endl;

  total = 0;

  for (int c = 0; c < conv0->shape(1); c++)
  {
    for (int h = 0; h < conv0->shape(2); h++)
    {
      for (int w = 0; w < conv0->shape(3); w++)
      {
        total += conv0->cpu_data()[conv0->offset(0, c, h, w)];
      }
    }
  }
  std::cerr << "Matt, second total=" << total << std::endl;

  std::vector<caffe::Blob<float>*> outputVecs = m_Net->output_blobs();

  // Need to find output with correct size.
  int i = 0;
  total = 0;
  for (i = 0; i < outputVecs.size(); i++)
  {
    caffe::Blob<float>* blob = outputVecs[i];

    if (   blob->shape(0) == 1
        && blob->shape(1) == 2
        && blob->shape(2) == m_DownSampledOutputImage.rows
        && blob->shape(3) == m_DownSampledOutputImage.cols
       )
    {

      for (int h = 0; h < m_DownSampledOutputImage.rows; ++h)
      {
        for (int w = 0; w < m_DownSampledOutputImage.cols; ++w)
        {
          if (  blob->cpu_data()[blob->offset(0, 1, h, w)]  // channel 1 = p(foreground)
              > blob->cpu_data()[blob->offset(0, 0, h, w)]  // channel 2 = p(background)
              )
          {
            m_DownSampledOutputImage.at<unsigned char>(h, w) = 255;
            total += 1;
          }
          else
          {
            m_DownSampledOutputImage.at<unsigned char>(h, w) = 0;
          }
        }
      }
      MITK_INFO << "Matt, total added to output=" << total << ", channel=" << i;
      break;
    }
  }

  if (i == outputVecs.size())
  {
    mitkThrow() << "Failed to find correct output blob.";
  }

  // Should not re-allocate if m_ResizedInputImage the right size.
  cv::resize(m_DownSampledOutputImage, m_UpSampledOutputImage, cv::Size(inputImage->GetDimension(0), inputImage->GetDimension(1)), 0, 0, cv::INTER_NEAREST);

  // Copy to output. Relies on the fact that output is always 1 channel, 8 bit, uchar.
  mitk::ImageWriteAccessor writeAccess(outputImage);
  void* vPointer = writeAccess.GetData();
  memcpy(vPointer, m_UpSampledOutputImage.data, m_UpSampledOutputImage.rows * m_UpSampledOutputImage.cols);
}

} // end namespace

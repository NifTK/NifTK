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
#include <cv.h>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

namespace niftk
{

//-----------------------------------------------------------------------------
class CaffeFCNSegmentorPrivate {

public:

  CaffeFCNSegmentorPrivate(const std::string& networkDescriptionFileName,
                    const std::string& networkWeightsFileName,
                    const std::string& inputLayerName,
                    const std::string& outputBlobName
                   );
  virtual ~CaffeFCNSegmentorPrivate();

  void SetOffset(const std::vector<float>& offset);
  void Segment(const mitk::Image::Pointer& inputImage, const mitk::Image::Pointer& outputImage);

private:

  void ValidateInputs(const mitk::Image::Pointer& inputImage,
                      const mitk::Image::Pointer& outputImage);

  std::string                          m_InputLayerName;
  std::string                          m_OutputBlobName;
  std::unique_ptr<caffe::Net<float> >  m_Net;
  std::unique_ptr<caffe::Blob<float> > m_InputBlob;
  std::unique_ptr<caffe::Blob<float> > m_InputLabel;
  std::vector<float>                   m_Offset;
  int                                  m_NumberOfChannels;
  int                                  m_FloatType;
  cv::Mat                              m_ResizedInputImage;
  cv::Mat                              m_OffsetValueImage;
  cv::Mat                              m_InputImageWithOffset;
  cv::Mat                              m_DownSampledFloatImage;
  cv::Mat                              m_UpSampledFloatImage;
  cv::Mat                              m_ClassifiedImage;
  cv::Mat                              m_ResizedOutputImage;
};

//-----------------------------------------------------------------------------
CaffeFCNSegmentorPrivate::CaffeFCNSegmentorPrivate(const std::string& networkDescriptionFileName,
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

  m_NumberOfChannels = memoryLayer->channels();
  if (m_NumberOfChannels != 1 && m_NumberOfChannels != 3)
  {
    mitkThrow() << "Currently, only 1 or 3 channel RGB networks are supported.";
  }

  m_Offset.resize(m_NumberOfChannels);
  for (int i = 0; i < m_Offset.size(); i++)
  {
    m_Offset[i] = 0;
  }

  if (m_NumberOfChannels == 1)
  {
    m_FloatType = CV_32FC1;
    m_OffsetValueImage = cvCreateMat(memoryLayer->height(), memoryLayer->width(), m_FloatType);
    m_OffsetValueImage.setTo(m_Offset[0]);
  }
  else
  {
    m_FloatType = CV_32FC3;
    m_OffsetValueImage = cvCreateMat(memoryLayer->height(), memoryLayer->width(), m_FloatType);
    m_OffsetValueImage.setTo(cv::Vec3f(m_Offset[0], m_Offset[1], m_Offset[2]));
  }

  m_InputImageWithOffset = cvCreateMat(memoryLayer->height(), memoryLayer->width(), m_FloatType);
  m_InputBlob.reset(new caffe::Blob<float>(1, memoryLayer->channels(), memoryLayer->height(), memoryLayer->width()));
  m_InputLabel.reset(new caffe::Blob<float>(1, 1, 1, 1));
  m_InputLabel->mutable_cpu_data()[m_InputBlob->offset(0, 0, 0, 0)] = 1;

  MITK_INFO << "CaffeFCNSegmentorPrivate: Running in " << m_NumberOfChannels
            << " channel mode, with float type=" << m_FloatType;
}


//-----------------------------------------------------------------------------
CaffeFCNSegmentorPrivate::~CaffeFCNSegmentorPrivate()
{
  // m_Net destroyed by std::unique_ptr.
  // m_InputBlob destroyed by std::unique_ptr.
}


//-----------------------------------------------------------------------------
void CaffeFCNSegmentorPrivate::SetOffset(const std::vector<float>& offset)
{
  if (offset.size() != 1 && offset.size() != 3)
  {
    mitkThrow() << "Only 1 or 3 channels are supported.";
  }
  if (offset.size() != m_NumberOfChannels)
  {
    mitkThrow() << "Programming bug: offset size=" << offset.size()
                << ", channels=" << m_NumberOfChannels;
  }

  m_Offset = offset;
  if (m_NumberOfChannels == 1)
  {
    m_OffsetValueImage.setTo(m_Offset[0]);
    MITK_INFO << "CaffeFCNSegmentorPrivate: offset=" << m_Offset[0];
  }
  else
  {
    m_OffsetValueImage.setTo(cv::Vec3f(m_Offset[0], m_Offset[1], m_Offset[2]));
    MITK_INFO << "CaffeFCNSegmentorPrivate: offset=" <<
                 m_Offset[0] << ", " << m_Offset[1] << ", " << m_Offset[2];
  }
}


//-----------------------------------------------------------------------------
void CaffeFCNSegmentorPrivate::ValidateInputs(const mitk::Image::Pointer& inputImage,
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
void CaffeFCNSegmentorPrivate::Segment(const mitk::Image::Pointer& inputImage,
                                       const mitk::Image::Pointer& outputImage)
{

  this->ValidateInputs(inputImage, outputImage);

  boost::shared_ptr<caffe::MemoryDataLayer<float> > memoryLayer =
      boost::dynamic_pointer_cast <caffe::MemoryDataLayer<float> >(m_Net->layer_by_name(m_InputLayerName));

  cv::Mat wrappedImage = niftk::MitkImageToOpenCVMat(inputImage);
  cv::resize(wrappedImage, m_ResizedInputImage, cv::Size(memoryLayer->width(), memoryLayer->height()));
  cv::add(m_ResizedInputImage, m_OffsetValueImage, m_InputImageWithOffset, cv::noArray(), m_FloatType);

  // Copy data to Caffe blob.
  if (m_NumberOfChannels == 1)
  {
    for (int h = 0; h < m_InputImageWithOffset.rows; ++h)
    {
      for (int w = 0; w < m_InputImageWithOffset.cols; ++w)
      {
        m_InputBlob->mutable_cpu_data()[m_InputBlob->offset(0, 0, h, w)]
          = m_InputImageWithOffset.at<unsigned char>(h, w);
      }
    }
  }
  else
  {
    for (int c = 0; c < m_NumberOfChannels; ++c)
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
  }
  memoryLayer->Reset(m_InputBlob->mutable_cpu_data(), m_InputLabel->mutable_cpu_data(), 1);

  // Runs Caffe classification
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
  if (   m_DownSampledFloatImage.rows != outputBlob->shape(2)
      || m_DownSampledFloatImage.cols != outputBlob->shape(3)
      || m_DownSampledFloatImage.channels() != 2
      || m_DownSampledFloatImage.type() != CV_32FC2
      )
  {
    m_DownSampledFloatImage = cvCreateMat(outputBlob->shape(2), outputBlob->shape(3), CV_32FC2);
  }

  // Now fill the output float data.
  for (int c = 0; c < 2; c++)
  {
    for (int h = 0; h < m_DownSampledFloatImage.rows; ++h)
    {
      for (int w = 0; w < m_DownSampledFloatImage.cols; ++w)
      {
        m_DownSampledFloatImage.at<cv::Vec2f>(h, w)[c] =
          outputBlob->cpu_data()[outputBlob->offset(0, c, h, w)];
      }
    }
  }
  cv::resize(m_DownSampledFloatImage, m_UpSampledFloatImage, cv::Size(memoryLayer->width(), memoryLayer->height()));

  if (   m_ClassifiedImage.rows != memoryLayer->height()
      || m_ClassifiedImage.cols != memoryLayer->width()
      || m_ClassifiedImage.channels() != 1
      || m_ClassifiedImage.type() != CV_8UC1
      )
  {
    m_ClassifiedImage = cvCreateMat(memoryLayer->height(), memoryLayer->width(), CV_8UC1);
  }

  // Now copy data out.
  cv::Vec2f   v;
  m_ClassifiedImage.setTo(0);
  for (int h = 0; h < m_ClassifiedImage.rows; ++h)
  {
    for (int w = 0; w < m_ClassifiedImage.cols; ++w)
    {
      v = m_UpSampledFloatImage.at<cv::Vec2f>(h, w);
      if (v[1] > v[0])
      {
        m_ClassifiedImage.at<unsigned char>(h, w) = 255;
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

  // Should not keep re-allocating if m_ResizedOutputImage the right size.
  cv::resize(m_ClassifiedImage, m_ResizedOutputImage, cv::Size(wrappedImage.cols, wrappedImage.rows), 0, 0, CV_INTER_NN);

  // Copy to output. This relies on the fact that output is always 1 channel, 8 bit, uchar.
  mitk::ImageWriteAccessor writeAccess(outputImage);
  void* vPointer = writeAccess.GetData();
  memcpy(vPointer, m_ResizedOutputImage.data, m_ResizedOutputImage.rows * m_ResizedOutputImage.cols);
}


//-----------------------------------------------------------------------------
CaffeFCNSegmentor::CaffeFCNSegmentor(const std::string& networkDescriptionFileName,
                                     const std::string& networkWeightsFileName,
                                     const std::string& inputLayerName,
                                     const std::string& outputBlobName
                                    )
: m_Impl(new CaffeFCNSegmentorPrivate(networkDescriptionFileName,
                                      networkWeightsFileName,
                                      inputLayerName,
                                      outputBlobName
                                      ))
{
}


//-----------------------------------------------------------------------------
CaffeFCNSegmentor::~CaffeFCNSegmentor()
{
}


//-----------------------------------------------------------------------------
void CaffeFCNSegmentor::SetOffset(const std::vector<float>& offset)
{
  m_Impl->SetOffset(offset);
}


//-----------------------------------------------------------------------------
void CaffeFCNSegmentor::Segment(const mitk::Image::Pointer& inputImage,
                                const mitk::Image::Pointer& outputImage)
{
  m_Impl->Segment(inputImage, outputImage);
}

} // end namespace

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
#include <cv.h>
#ifdef _WIN32
#define GLOG_NO_ABBREVIATED_SEVERITIES
#pragma push_macro("STRICT")
#undef STRICT
#include <caffe/caffe.hpp>
#pragma pop_macro("STRICT")
#else
#include <caffe/caffe.hpp>
#endif
#include <caffe/layers/memory_data_layer.hpp>


namespace niftk
{

//-----------------------------------------------------------------------------
class CaffeFCNSegmentorPrivate {

public:

  CaffeFCNSegmentorPrivate(const std::string& networkDescriptionFileName,
                    const std::string& networkWeightsFileName,
                    const std::string& inputLayerName,
                    const std::string& outputBlobName,
                    const int& gpuDevice
                   );
  virtual ~CaffeFCNSegmentorPrivate();

  void Segment(const mitk::Image::Pointer& inputImage, const mitk::Image::Pointer& outputImage);

private:

  void ValidateInputs(const mitk::Image::Pointer& inputImage,
                      const mitk::Image::Pointer& outputImage);

  std::string                          m_InputLayerName;
  std::string                          m_OutputBlobName;
  std::unique_ptr<caffe::Net<float> >  m_Net;
  cv::Mat                              m_TransposedInputImage;
  cv::Mat                              m_ResizedInputImage;
  cv::Mat                              m_DownSampledFloatImage;
  cv::Mat                              m_UpSampledFloatImage;
  cv::Mat                              m_ClassifiedImage;
  cv::Mat                              m_TransposedOutputImage;
  cv::Mat                              m_ResizedOutputImage;
};

//-----------------------------------------------------------------------------
CaffeFCNSegmentorPrivate::CaffeFCNSegmentorPrivate(const std::string& networkDescriptionFileName,
                                                   const std::string& networkWeightsFileName,
                                                   const std::string& inputLayerName,
                                                   const std::string& outputBlobName,
                                                   const int& gpuDevice
                                                  )
: m_InputLayerName(inputLayerName)
, m_OutputBlobName(outputBlobName)
, m_Net(nullptr)
{
  if (networkDescriptionFileName.empty())
  {
    mitkThrow() << "Caffe network description filename (.prototxt) is empty.";
  }
  if (networkWeightsFileName.empty())
  {
    mitkThrow() << "Caffe network weights filename (.caffemodel) is empty.";
  }

  caffe::Caffe::set_mode(caffe::Caffe::CPU);

#ifndef _USE_CUDA
  if (gpuDevice >= 0)
  {
    MITK_INFO << "GPU device " << gpuDevice << " is requested, but ignored, as CUDA is not compiled in.";
  }
#endif

#ifdef _USE_CUDA

  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));

  if (gpuDevice < 0)
  {
    MITK_WARN << "Requested GPU device number is negative, so switching to CPU mode.";
  }
  else if (gpuDevice >= count)
  {
    MITK_WARN << "Requested GPU device number (" << gpuDevice
              << ") is too high, given the number of GPU devices (" << count << ")";
  }
  else
  {
    caffe::Caffe::SetDevice(gpuDevice); // we are only allowing for 1 device at the moment.
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
  }
#endif

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
}


//-----------------------------------------------------------------------------
CaffeFCNSegmentorPrivate::~CaffeFCNSegmentorPrivate()
{
  // m_Net destroyed by std::unique_ptr.
  // m_InputBlob destroyed by std::unique_ptr.
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
  cv::transpose(wrappedImage, m_TransposedInputImage);
  cv::resize(m_TransposedInputImage, m_ResizedInputImage, cv::Size(memoryLayer->width(), memoryLayer->height()));

  std::vector<cv::Mat> dv;
  dv.push_back(m_ResizedInputImage);

  std::vector<int> dvl;
  dvl.push_back(1);

  memoryLayer->AddMatVector(dv,dvl);

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

  cv::resize(m_DownSampledFloatImage, m_UpSampledFloatImage, cv::Size(memoryLayer->width(), memoryLayer->height()), 0, 0, CV_INTER_CUBIC);

  if (   m_ClassifiedImage.rows != m_UpSampledFloatImage.rows
      || m_ClassifiedImage.cols != m_UpSampledFloatImage.cols
      || m_ClassifiedImage.channels() != 1
      || m_ClassifiedImage.type() != CV_8UC1
      )
  {
    m_ClassifiedImage = cvCreateMat(m_UpSampledFloatImage.rows, m_UpSampledFloatImage.cols, CV_8UC1);
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

  // Should not keep re-allocating if m_TransposedOutputImage or m_ResizedOutputImage the right size.
  cv::transpose(m_ClassifiedImage, m_TransposedOutputImage);
  cv::resize(m_TransposedOutputImage, m_ResizedOutputImage, cv::Size(wrappedImage.cols, wrappedImage.rows), 0, 0, CV_INTER_NN);

  // Copy to output. This relies on the fact that output is always 1 channel, 8 bit, uchar.
  mitk::ImageWriteAccessor writeAccess(outputImage);
  void* vPointer = writeAccess.GetData();
  memcpy(vPointer, m_ResizedOutputImage.data, m_ResizedOutputImage.rows * m_ResizedOutputImage.cols);
  outputImage->Modified();
}


//-----------------------------------------------------------------------------
CaffeFCNSegmentor::CaffeFCNSegmentor(const std::string& networkDescriptionFileName,
                                     const std::string& networkWeightsFileName,
                                     const std::string& inputLayerName,
                                     const std::string& outputBlobName,
                                     const int& gpuDevice
                                    )
: m_Impl(new CaffeFCNSegmentorPrivate(networkDescriptionFileName,
                                      networkWeightsFileName,
                                      inputLayerName,
                                      outputBlobName,
                                      gpuDevice
                                      ))
{
}


//-----------------------------------------------------------------------------
CaffeFCNSegmentor::CaffeFCNSegmentor(const std::string& networkDescriptionFileName,  // Purposefully hidden.
                                     const std::string& networkWeightsFileName
                                    )
: m_Impl(new CaffeFCNSegmentorPrivate(networkDescriptionFileName, networkWeightsFileName,
                                      "data", "prediction", 0)) // only uses GPU if CUDA compiled in.
{
}


//-----------------------------------------------------------------------------
CaffeFCNSegmentor::~CaffeFCNSegmentor()
{
}


//-----------------------------------------------------------------------------
void CaffeFCNSegmentor::Segment(const mitk::Image::Pointer& inputImage,
                                const mitk::Image::Pointer& outputImage)
{
  m_Impl->Segment(inputImage, outputImage);
}

} // end namespace

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkCaffeFCNSegmentor_h
#define niftkCaffeFCNSegmentor_h

#include "niftkCaffeExports.h"
#include <itkObject.h>
#include <itkObjectFactoryBase.h>
#include <mitkCommon.h>
#include <mitkDataNode.h>
#include <mitkImage.h>
#include <memory>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <opencv/cv.h>

namespace niftk
{

/**
 * \class CaffeFCNSegmentor
 * \brief RAII class to coordinate Caffe FCN-based segmentation.
 *
 * All errors must be thrown as a subclass of mitk::Exception.
 *
 * The class is stateful. It is assumed that the constructor will
 * initialise the network, so if the constructor is successful,
 * the class should be ready to segment. Then all subsequent
 * calls to the Segment method will compute a segmented image.
 */
class NIFTKCAFFE_EXPORT CaffeFCNSegmentor : public itk::Object
{
public:

  mitkClassMacroItkParent(CaffeFCNSegmentor, itk::Object)
  mitkNewMacro6Param(CaffeFCNSegmentor, const std::string&, const std::string&, const mitk::Vector3D&, const mitk::Point2I&, const std::string&, const mitk::Point2I&)

  /**
   * \brief Segments the inputImage, and writes to outputImage.
   * \param inputImage RGB or RGBA image
   * \param outputImage grey scale (single channel), 8 bit, unsigned char image.
   */
  void Segment(const mitk::Image::Pointer& inputImage,
               const mitk::Image::Pointer& outputImage
               );

protected:

  /**
   * \brief Constructor
   * \param networkDescriptionFileName .prototxt file
   * \param networkWeightsFileName .caffemodel file
   * \param meanRGB additive offset, normally a mean value from the training process
   * \param inputNetworkImageSize image size that the network expects
   * \param outputLayerName name of the Caffe layer representing the output
   * \param outputNetworkImageSize image size that the network outputs
   */
  CaffeFCNSegmentor(const std::string&    networkDescriptionFileName,  // Purposefully hidden.
                    const std::string&    networkWeightsFileName,
                    const mitk::Vector3D& offsetRGB,
                    const mitk::Point2I&  inputNetworkImageSize,
                    const std::string&    outputLayerName,
                    const mitk::Point2I&  outputNetworkImageSize
                   );
  virtual ~CaffeFCNSegmentor();                                        // Purposefully hidden.

  CaffeFCNSegmentor(const CaffeFCNSegmentor&);                         // Purposefully not implemented.
  CaffeFCNSegmentor& operator=(const CaffeFCNSegmentor&);              // Purposefully not implemented.

private:

  void ValidateInputs(const mitk::Image::Pointer& inputImage,
                      const mitk::Image::Pointer& outputImage);

  boost::shared_ptr<caffe::MemoryDataLayer<float> > GetAndValidateMemoryLayer();

  mitk::Vector3D                       m_OffsetRGB;
  mitk::Point2I                        m_InputNetworkImageSize;
  std::string                          m_OutputLayerName;
  mitk::Point2I                        m_OutputNetworkImageSize;
  std::unique_ptr<caffe::Net<float> >  m_Net;
  std::unique_ptr<caffe::Blob<float> > m_InputBlob;
  std::unique_ptr<caffe::Blob<float> > m_InputLabel;
  cv::Mat                              m_ResizedInputImage;
  cv::Mat                              m_OffsetValueImage;
  cv::Mat                              m_InputImageWithOffset;
  cv::Mat                              m_DownSampledOutputImage;
  cv::Mat                              m_UpSampledOutputImage;
}; // end class

} // end namespace

#endif

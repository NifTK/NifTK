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
#include <cv.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>
#include <mitkCommon.h>
#include <mitkDataNode.h>
#include <mitkImage.h>
#include <memory>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

namespace niftk
{

/**
 * \class CaffeFCNSegmentor
 * \brief RAII class to coordinate Caffe FCN semantic segmentation.
 *
 * All errors must be thrown as a subclass of mitk::Exception.
 *
 * The class is stateful. It is assumed that the constructor will
 * initialise the network, so if the constructor is successful,
 * the class should be ready to segment. Then all subsequent
 * calls to the Segment method will compute a segmented image.
 *
 * This class assumes the existence of a Caffe MemoryData layer
 * meaning that the images are loaded from in-memory.
 * So, in the Segment() method, the inputImage is loaded into
 * the MemoryData layer. The outputImage is the already-created
 * unsigned char binary mask, which must be exactly the same size
 * as the inputImage.
 */
class NIFTKCAFFE_EXPORT CaffeFCNSegmentor : public itk::Object
{
public:

  mitkClassMacroItkParent(CaffeFCNSegmentor, itk::Object)
  mitkNewMacro4Param(CaffeFCNSegmentor, const std::string&, const std::string&, const std::string&, const std::string&)

  /**
   * \brief Segments the inputImage, and writes to outputImage.
   * \param inputImage RGB or RGBA vector image, or grey scale (single channel) scalar image
   * \param outputImage grey scale (single channel), 8 bit, unsigned char image, with 2 values, 0 = background, 1 = foreground.
   */
  void Segment(const mitk::Image::Pointer& inputImage,
               const mitk::Image::Pointer& outputImage
              );

protected:

  /**
   * \brief Constructor
   * \param networkDescriptionFileName .prototxt file
   * \param networkWeightsFileName .caffemodel file
   */
  CaffeFCNSegmentor(const std::string& networkDescriptionFileName,  // Purposefully hidden.
                    const std::string& networkWeightsFileName,
                    const std::string& inputLayerName,
                    const std::string& outputBlobName
                   );
  virtual ~CaffeFCNSegmentor();                                        // Purposefully hidden.

  CaffeFCNSegmentor(const CaffeFCNSegmentor&);                         // Purposefully not implemented.
  CaffeFCNSegmentor& operator=(const CaffeFCNSegmentor&);              // Purposefully not implemented.

private:

  void ValidateInputs(const mitk::Image::Pointer& inputImage,
                      const mitk::Image::Pointer& outputImage);

  std::string                          m_InputLayerName;
  std::string                          m_OutputBlobName;
  std::unique_ptr<caffe::Net<float> >  m_Net;
  std::unique_ptr<caffe::Blob<float> > m_InputBlob;
  std::unique_ptr<caffe::Blob<float> > m_InputLabel;
  mitk::Vector3D                       m_OffsetRGB;
  mitk::Point2I                        m_Margin;
  cv::Mat                              m_CroppedInputImage;
  cv::Mat                              m_ResizedInputImage;
  cv::Mat                              m_OffsetValueImage;
  cv::Mat                              m_InputImageWithOffset;
  cv::Mat                              m_DownSampledOutputImage;
  cv::Mat                              m_UpSampledOutputImage;
  cv::Mat                              m_UpSampledPaddedOutputImage;
  cv::Mat                              m_ResizedOutputImage;
}; // end class

} // end namespace

#endif

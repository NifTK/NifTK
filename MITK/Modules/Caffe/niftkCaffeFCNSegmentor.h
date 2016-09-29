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

namespace niftk
{

class CaffeFCNSegmentorPrivate;

/**
 * \class CaffeFCNSegmentor
 * \brief RAII class to coordinate Caffe FCN semantic segmentation.
 *
 * This class performs simple, background=0, foreground=255 classification.
 * This class expects the Caffe network to have a single output node,
 * containing probabilities channel 0 = p(background), channel 1 =
 * p(foreground) and then this class will simply iterate through
 * each pixel, and set the value 255 if p(forground) > p(background).
 *
 * This class assumes the existence of a Caffe MemoryData layer
 * in the model architecture, so images are loaded from memory.
 * The outputImage is assumed to be the already-created
 * unsigned char binary mask, which must be exactly the same size
 * as the inputImage.
 *
 * The class is stateful. It is assumed that the constructor will
 * initialise the network, so if the constructor is successful,
 * the class should be ready to segment. Then all subsequent
 * calls to the Segment method will compute a segmented image.
 *
 * All errors must be thrown as a subclass of mitk::Exception.
 */
class NIFTKCAFFE_EXPORT CaffeFCNSegmentor : public itk::Object
{
public:

  mitkClassMacroItkParent(CaffeFCNSegmentor, itk::Object)
  mitkNewMacro4Param(CaffeFCNSegmentor, const std::string&, const std::string&, const std::string&, const std::string&)

  void SetRGBOffset(const mitk::Vector3D& offset);

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
   * \param inputLayerName name of an input MemoryData layer, e.g. "data".
   * \param outputBlobName name of an output blob, containing 2 channel, float, of probabilities.
   */
  CaffeFCNSegmentor(const std::string& networkDescriptionFileName,  // Purposefully hidden.
                    const std::string& networkWeightsFileName,
                    const std::string& inputLayerName,
                    const std::string& outputBlobName
                   );
  virtual ~CaffeFCNSegmentor();                                     // Purposefully hidden.

  CaffeFCNSegmentor(const CaffeFCNSegmentor&);                      // Purposefully not implemented.
  CaffeFCNSegmentor& operator=(const CaffeFCNSegmentor&);           // Purposefully not implemented.

private:

  std::unique_ptr<CaffeFCNSegmentorPrivate> m_Impl;

}; // end class

} // end namespace

#endif

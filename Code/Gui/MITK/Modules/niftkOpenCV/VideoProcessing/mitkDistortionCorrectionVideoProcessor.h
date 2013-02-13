/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKDISTORTIONCORRECTIONVIDEOPROCESSOR_H
#define MITKDISTORTIONCORRECTIONVIDEOPROCESSOR_H

#include "niftkOpenCVExports.h"
#include "mitkVideoProcessorTemplateMethod.h"

/**
 * class VideoProcessorTemplateMethod
 * \brief Uses Template method to read from a capture device, do some processing,
 * and write to a cvVideoWriter object.
 */
namespace mitk {

class NIFTKOPENCV_EXPORT DistortionCorrectionVideoProcessor : public mitk::VideoProcessorTemplateMethod
{

public:

  mitkClassMacro(DistortionCorrectionVideoProcessor, mitk::VideoProcessorTemplateMethod);
  mitkNewMacro4Param(DistortionCorrectionVideoProcessor, const std::string&, const std::string&, const CvMat&, const CvMat&);

protected:

  ~DistortionCorrectionVideoProcessor();
  DistortionCorrectionVideoProcessor(CvCapture *capture, CvVideoWriter *writer);
  DistortionCorrectionVideoProcessor(const std::string&, const std::string&, const CvMat&, const CvMat&);

  DistortionCorrectionVideoProcessor(const VideoProcessorTemplateMethod&); // Purposefully not implemented.
  DistortionCorrectionVideoProcessor& operator=(const VideoProcessorTemplateMethod&); // Purposefully not implemented.

  virtual void DoProcessing(const IplImage &input, IplImage &output);

private:

  CvMat *m_Intrinsic;
  CvMat *m_Distortion;
  IplImage *m_MapX;
  IplImage *m_MapY;

}; // end class

} // end namespace

#endif // MITKDISTORTIONCORRECTIONVIDEOPROCESSOR_H

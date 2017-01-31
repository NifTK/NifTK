/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkVideoWriter_h
#define mitkVideoWriter_h

#include <cv.h>
#include <highgui.h>
#include <cstdlib>
#include <iostream>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

/**
 * \class VideoWriter
 * \brief Class to handle video writing for the openCV module, not exported.
 */
namespace mitk {

class VideoWriter : public itk::Object
{

public:

  mitkClassMacroItkParent(VideoWriter, itk::Object)

  /**
   * \brief This method initialises the opencv video writer.
   */
  void Initialize(std::string filename, double framesPerSecond, cv::Size imageSize );

  /**
   * \brief This method writes a frame of video
   */
  void WriteFrame ( IplImage image);

protected:

  ~VideoWriter();

  VideoWriter(const VideoWriter&); // Purposefully not implemented.
  VideoWriter& operator=(const VideoWriter&); // Purposefully not implemented.

  /**
   * \brief Returns the writer,
   */
  CvVideoWriter* GetWriter() const;

private:

  CvVideoWriter *m_Writer;
  int           m_FourCC;  // the codec to use
  bool          m_IsColour; //is is colour
}; // end class

} // end namespace

#endif // MITKVIDEOWRITER_H

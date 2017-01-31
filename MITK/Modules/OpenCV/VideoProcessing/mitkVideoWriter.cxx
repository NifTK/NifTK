/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkVideoWriter.h"

namespace mitk
{

//-----------------------------------------------------------------------------
VideoWriter::~VideoWriter()
{
  if ( m_Writer != NULL )
  {
    cvReleaseVideoWriter(&m_Writer);
  }
}

//-----------------------------------------------------------------------------
VideoWriter::VideoWriter()
: m_Writer(NULL)
, m_FourCC(CV_FOURCC('M','J','P','G')
, m_IsColour(true)
{
}

//-----------------------------------------------------------------------------
void VideoWriter::Initialize(std::string filename, double framesPerSecond, cv::Size imageSize)
{
  if (m_Writer == NULL)
  {
    m_Writer = cvCreateVideoWriter(filename.c_str(), m_FourCC, framesPerSecond, imageSize, m_IsColour);
    if (m_Writer == NULL)
    {
      throw std::logic_error("Could not create video writer.");
    }
  }
}

//-----------------------------------------------------------------------------
CvVideoWriter* VideoWriter::GetWriter() const
{
  if (m_Writer == NULL)
  {
    throw std::logic_error("The CvVideoWriter is NULL");
  }
  return m_Writer;
}

//-----------------------------------------------------------------------------
void VideoWriter::WriteFrame(IplImage image)
{
  if (m_Writer == NULL)
  {
    cvWriteFrame(m_Writer,&image);
  }
  else
  {
    throw std::logic_error("No video writer.");
  }
}


} // end namespace

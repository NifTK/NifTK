/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-07-25 07:31:59 +0100 (Wed, 25 Jul 2012) $
 Revision          : $Revision: 9401 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkIGIOpenCVDataSource.h"
#include "mitkIGIOpenCVDataType.h"

//-----------------------------------------------------------------------------
QmitkIGIOpenCVDataSource::QmitkIGIOpenCVDataSource()
{
  m_VideoSource = mitk::OpenCVVideoSource::New();
  m_VideoSource->SetVideoCameraInput(0);

  this->SetName("Video");
  this->SetType("Frame Grabber");
  this->SetDescription("OpenCV");
  this->SetStatus("Initialised");
}


//-----------------------------------------------------------------------------
QmitkIGIOpenCVDataSource::~QmitkIGIOpenCVDataSource()
{
  this->StopCapturing();
}


//-----------------------------------------------------------------------------
mitk::OpenCVVideoSource* QmitkIGIOpenCVDataSource::GetVideoSource() const
{
  return m_VideoSource;
}


//-----------------------------------------------------------------------------
bool QmitkIGIOpenCVDataSource::CanHandleData(mitk::IGIDataType* data) const
{
  bool result = false;
  if (static_cast<mitk::IGIOpenCVDataType*>(data) != NULL)
  {
    result = true;
  }
  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGIOpenCVDataSource::StartCapturing()
{
  if (m_VideoSource.IsNotNull() && !m_VideoSource->IsCapturingEnabled())
  {
    m_VideoSource->StartCapturing();
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIOpenCVDataSource::StopCapturing()
{
  if (m_VideoSource.IsNotNull() && m_VideoSource->IsCapturingEnabled())
  {
    m_VideoSource->StopCapturing();
  }
}

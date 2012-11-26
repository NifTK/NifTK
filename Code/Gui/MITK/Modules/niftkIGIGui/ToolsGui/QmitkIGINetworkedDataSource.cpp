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

#include "QmitkIGINetworkedDataSource.h"

//-----------------------------------------------------------------------------
QmitkIGINetworkDataSource::QmitkIGINetworkDataSource()
: m_Socket(NULL)
, m_ClientDescriptor(NULL)
{
}


//-----------------------------------------------------------------------------
QmitkIGINetworkDataSource::~QmitkIGINetworkDataSource()
{
  if (m_Socket != NULL)
  {
    delete m_Socket;
  }
  if (m_ClientDescriptor != NULL)
  {
    delete m_ClientDescriptor;
  }
}


//-----------------------------------------------------------------------------
int QmitkIGINetworkDataSource::GetPort() const
{
  int result = -1;
  if (m_Socket != NULL)
  {
    result = m_Socket->getPort();
  }
  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGINetworkDataSource::SendMessage(OIGTLMessage::Pointer msg)
{
  if (m_Socket != NULL)
  {
    m_Socket->sendMessage(msg);
  }
}

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

#ifndef QMITKIGINIFTYLINKDATATYPE_H
#define QMITKIGINIFTYLINKDATATYPE_H

#include "niftkIGIGuiExports.h"
#include "mitkIGIDataType.h"
#include "OIGTLMessage.h"

/**
 * \class QmitkIGINiftyLinkDataType
 * \brief Data wrapper for messages coming from NiftyLink.
 */
class NIFTKIGIGUI_EXPORT QmitkIGINiftyLinkDataType : public mitk::IGIDataType
{

public:

  mitkClassMacro(QmitkIGINiftyLinkDataType, mitk::IGIDataType);
  itkNewMacro(QmitkIGINiftyLinkDataType);

  virtual void* GetData() const { return m_Message.data(); }

  OIGTLMessage* GetMessage() const { return m_Message.data(); }
  void SetMessage(OIGTLMessage* message) { m_Message = message; this->Modified(); }

protected:

  QmitkIGINiftyLinkDataType(); // Purposefully hidden.
  virtual ~QmitkIGINiftyLinkDataType(); // Purposefully hidden.

  QmitkIGINiftyLinkDataType(const QmitkIGINiftyLinkDataType&); // Purposefully not implemented.
  QmitkIGINiftyLinkDataType& operator=(const QmitkIGINiftyLinkDataType&); // Purposefully not implemented.

private:

  OIGTLMessage::Pointer m_Message;

}; // end class

#endif

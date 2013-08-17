/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkIGINiftyLinkDataType_h
#define QmitkIGINiftyLinkDataType_h

#include "niftkIGIGuiExports.h"
#include <mitkIGIDataType.h>
#include <NiftyLinkMessage.h>

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

  NiftyLinkMessage* GetMessage() const { return m_Message.data(); }
  void SetMessage(NiftyLinkMessage* message) { m_Message = message; this->Modified(); }

protected:

  QmitkIGINiftyLinkDataType(); // Purposefully hidden.
  virtual ~QmitkIGINiftyLinkDataType(); // Purposefully hidden.

  QmitkIGINiftyLinkDataType(const QmitkIGINiftyLinkDataType&); // Purposefully not implemented.
  QmitkIGINiftyLinkDataType& operator=(const QmitkIGINiftyLinkDataType&); // Purposefully not implemented.

private:

  NiftyLinkMessage::Pointer m_Message;

}; // end class

#endif

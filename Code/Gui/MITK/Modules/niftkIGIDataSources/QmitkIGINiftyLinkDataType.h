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

#include "niftkIGIDataSourcesExports.h"
#include <mitkIGIDataType.h>
#include <MessageHandling/NiftyLinkMessageContainer.h>

/**
 * \class QmitkIGINiftyLinkDataType
 * \brief Data wrapper for messages coming from NiftyLink.
 */
class NIFTKIGIDATASOURCES_EXPORT QmitkIGINiftyLinkDataType : public mitk::IGIDataType
{

public:

  mitkClassMacro(QmitkIGINiftyLinkDataType, mitk::IGIDataType);
  itkNewMacro(QmitkIGINiftyLinkDataType);

  virtual void* GetData() const { return m_Message.data(); }

  niftk::NiftyLinkMessageContainer::Pointer GetMessageContainer() const { return m_Message; }
  void SetMessageContainer(niftk::NiftyLinkMessageContainer::Pointer message) { m_Message = message; this->Modified(); }

protected:

  QmitkIGINiftyLinkDataType(); // Purposefully hidden.
  virtual ~QmitkIGINiftyLinkDataType(); // Purposefully hidden.

  QmitkIGINiftyLinkDataType(const QmitkIGINiftyLinkDataType&); // Purposefully not implemented.
  QmitkIGINiftyLinkDataType& operator=(const QmitkIGINiftyLinkDataType&); // Purposefully not implemented.

private:

  niftk::NiftyLinkMessageContainer::Pointer m_Message;

}; // end class

#endif

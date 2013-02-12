/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGINiftyLinkDataSourceGui.h"
#include <Common/NiftyLinkXMLBuilder.h>

//-----------------------------------------------------------------------------
QmitkIGINiftyLinkDataSourceGui::QmitkIGINiftyLinkDataSourceGui()
{

}


//-----------------------------------------------------------------------------
QmitkIGINiftyLinkDataSourceGui::~QmitkIGINiftyLinkDataSourceGui()
{
}


//-----------------------------------------------------------------------------
QmitkIGINiftyLinkDataSource* QmitkIGINiftyLinkDataSourceGui::GetQmitkIGINiftyLinkDataSource()
{
  QmitkIGINiftyLinkDataSource* result = NULL;
  mitk::IGIDataSource* source = this->GetSource();
  if (source != NULL)
  {
    result = dynamic_cast<QmitkIGINiftyLinkDataSource*>(source);
  }
  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGINiftyLinkDataSourceGui::Initialize(QWidget *parent)
{
  QmitkIGINiftyLinkDataSource *source = this->GetQmitkIGINiftyLinkDataSource();

  if (source != NULL)
  {
    ClientDescriptorXMLBuilder* config = source->GetClientDescriptor();

    // Derived classes should make sure they can cope with a null config.
    this->Initialize(parent, config);
  }
}

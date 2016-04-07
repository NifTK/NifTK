/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIConfigurationDialog_h
#define niftkIGIConfigurationDialog_h

#include <niftkIGIDataSourcesExports.h>
#include <niftkIGIDataSourceI.h>

#include <QMap>
#include <QVariant>
#include <QString>
#include <QDialog>

namespace niftk
{

/**
* \class IGIConfigurationDialog
* \brief Used to send parameters to and from the IGIDataSourceServiceI at runtime.
*/
class NIFTKIGIDATASOURCES_EXPORT IGIConfigurationDialog : public QDialog
{
  Q_OBJECT

public:
  IGIConfigurationDialog(QWidget *parent, niftk::IGIDataSourceI::Pointer service);
  ~IGIConfigurationDialog();

protected:
  niftk::IGIDataSourceI::Pointer m_Service;
};

} // end namespace

#endif

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIInitialisationDialog_h
#define niftkIGIInitialisationDialog_h

#include <niftkIGIDataSourcesExports.h>

#include <QMap>
#include <QVariant>
#include <QString>
#include <QDialog>

namespace niftk
{

/**
* \class IGIInitialisationDialog
* \brief Initialisation dialogs must export their properties on completion.
*
* So, when the OK button is clicked, we can call GetProperties()
* to to retrieve all the data values, prior to creating the data source.
*/
class NIFTKIGIDATASOURCES_EXPORT IGIInitialisationDialog : public QDialog
{
  Q_OBJECT

public:
  IGIInitialisationDialog(QWidget *parent);
  ~IGIInitialisationDialog();

  virtual QMap<QString, QVariant> GetProperties() const;

protected:

  QMap<QString, QVariant> m_Properties;
};

} // end namespace

#endif

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkDataSourcesViewPreferencePage_h
#define niftkDataSourcesViewPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>
#include <QString>
#include <QColor>

class QWidget;
class QSpinBox;
class ctkDirectoryButton;

namespace niftk
{

/**
* \class DataSourcesViewPreferencePage
* \brief Preferences page for the surgical guidance plugin with choices for colours, default paths to save data etc.
* \ingroup uk_ac_ucl_cmic_igidatasources_internal
*/
class DataSourcesViewPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  DataSourcesViewPreferencePage();
  DataSourcesViewPreferencePage(const DataSourcesViewPreferencePage& other);
  ~DataSourcesViewPreferencePage();

  void Init(berry::IWorkbench::Pointer workbench) override;
  void CreateQtControl(QWidget* widget) override;
  QWidget* GetQtControl() const override;

  /**
  * \see IPreferencePage::PerformOk()
  */
  virtual bool PerformOk() override;

  /**
  * \see IPreferencePage::PerformCancel()
  */
  virtual void PerformCancel() override;

  /**
  * \see IPreferencePage::Update()
  */
  virtual void Update() override;

private:

  QWidget                      *m_MainControl;
  QSpinBox                     *m_FramesPerSecondSpinBox;
  ctkDirectoryButton           *m_DirectoryPrefix;
  bool                          m_Initializing;
  berry::IPreferences::Pointer  m_DataSourcesViewPreferencesNode;
};

} // end namespace

#endif


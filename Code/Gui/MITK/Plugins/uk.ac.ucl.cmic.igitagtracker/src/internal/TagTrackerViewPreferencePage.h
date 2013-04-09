/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef TagTrackerViewPreferencePage_h
#define TagTrackerViewPreferencePage_h

#include "berryIQtPreferencePage.h"
#include <berryIPreferences.h>
#include <QString>

class QWidget;
class QPushButton;

/**
 * \class TagTrackerViewPreferencePage
 * \brief Preferences page for the Tag Tracker View plugin.
 * \ingroup uk_ac_ucl_cmic_igitagtracker_internal
 *
 */
class TagTrackerViewPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  /// \brief Stores the name of the preferences node.
  static const std::string PREFERENCES_NODE_NAME;

  TagTrackerViewPreferencePage();
  TagTrackerViewPreferencePage(const TagTrackerViewPreferencePage& other);
  ~TagTrackerViewPreferencePage();

  void Init(berry::IWorkbench::Pointer workbench);

  void CreateQtControl(QWidget* widget);

  QWidget* GetQtControl() const;

  /**
   * \see IPreferencePage::PerformOk()
   */
  virtual bool PerformOk();

  /**
   * \see IPreferencePage::PerformCancel()
   */
  virtual void PerformCancel();

  /**
   * \see IPreferencePage::Update()
   */
  virtual void Update();

private slots:

private:

  QWidget        *m_MainControl;
  QPushButton    *m_DummyButton;
  bool            m_Initializing;

  berry::IPreferences::Pointer m_TagTrackerViewPreferencesNode;
};

#endif // TagTrackerViewPreferencePage_h


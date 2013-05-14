/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef PointRegViewPreferencePage_h
#define PointRegViewPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>
#include <QString>

class QWidget;
class QPushButton;

/**
 * \class PointRegViewPreferencePage
 * \brief Preferences page for the Point Based Registration View plugin.
 * \ingroup uk_ac_ucl_cmic_igitrackedpointer_internal
 *
 */
class PointRegViewPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  /// \brief Stores the name of the preferences node.
  static const std::string PREFERENCES_NODE_NAME;

  PointRegViewPreferencePage();
  PointRegViewPreferencePage(const PointRegViewPreferencePage& other);
  ~PointRegViewPreferencePage();

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

  berry::IPreferences::Pointer m_PointRegViewPreferencesNode;
};

#endif // PointRegViewPreferencePage_h


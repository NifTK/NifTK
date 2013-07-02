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
class QCheckBox;

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

  /**
   * \brief Stores the name of the preferences node.
   */
  static const std::string PREFERENCES_NODE_NAME;

  /**
   * \brief Stores the name of the preference node that contains the value of the Use ICP preference.
   */
  static const std::string USE_ICP_INITIALISATION;

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
  QCheckBox      *m_UseICPInitialisation;
  bool            m_Initializing;

  berry::IPreferences::Pointer m_PointRegViewPreferencesNode;
};

#endif // PointRegViewPreferencePage_h


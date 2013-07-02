/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef TrackedPointerViewPreferencePage_h
#define TrackedPointerViewPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>
#include <QString>

class QWidget;
class QCheckBox;
class ctkPathLineEdit;

/**
 * \class TrackedPointerViewPreferencePage
 * \brief Preferences page for the Tracked Pointer View plugin.
 * \ingroup uk_ac_ucl_cmic_igitrackedpointer_internal
 *
 */
class TrackedPointerViewPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  /**
   * \brief Stores the name of the preferences node.
   */
  static const std::string PREFERENCES_NODE_NAME;

  /**
   * \brief Stores the name of the preference node that contains the name of the calibration file.
   */
  static const std::string CALIBRATION_FILE_NAME;

  /**
   * \brief Stores the name of the preference node that stores the boolean of whether to update the ortho-view focus point.
   */
  static const std::string UPDATE_VIEW_COORDINATE_NAME;

  TrackedPointerViewPreferencePage();
  TrackedPointerViewPreferencePage(const TrackedPointerViewPreferencePage& other);
  ~TrackedPointerViewPreferencePage();

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

  QWidget         *m_MainControl;
  ctkPathLineEdit *m_CalibrationFileName;
  QCheckBox       *m_UpdateViewCoordinate;
  bool             m_Initializing;

  berry::IPreferences::Pointer m_TrackedPointerViewPreferencesNode;
};

#endif // TrackedPointerViewPreferencePage_h


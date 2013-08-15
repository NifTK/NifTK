/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef TrackedImageViewPreferencePage_h
#define TrackedImageViewPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>
#include <QString>

class QWidget;
class QPushButton;
class QDoubleSpinBox;
class ctkPathLineEdit;

/**
 * \class TrackedImageViewPreferencePage
 * \brief Preferences page for the Tracked Image View plugin.
 * \ingroup uk_ac_ucl_cmic_igitrackedimage_internal
 *
 */
class TrackedImageViewPreferencePage : public QObject, public berry::IQtPreferencePage
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
   * \brief Stores the name of the preference node that contains the image scaling in the x direction.
   */
  static const std::string X_SCALING;

  /**
   * \brief Stores the name of the preference node that contains the image scaling in the y direction.
   */
  static const std::string Y_SCALING;

  TrackedImageViewPreferencePage();
  TrackedImageViewPreferencePage(const TrackedImageViewPreferencePage& other);
  ~TrackedImageViewPreferencePage();

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
  bool             m_Initializing;
  QDoubleSpinBox  *m_XScaling;
  QDoubleSpinBox  *m_YScaling;

  berry::IPreferences::Pointer m_TrackedImageViewPreferencesNode;
};

#endif // TrackedImageViewPreferencePage_h


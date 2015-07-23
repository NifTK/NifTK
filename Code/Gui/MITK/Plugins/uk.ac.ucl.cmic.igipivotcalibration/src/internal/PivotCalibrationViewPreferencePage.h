/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef PivotCalibrationViewPreferencePage_h
#define PivotCalibrationViewPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>
#include <QString>
#include <ctkPathLineEdit.h>

class QWidget;
class QCheckBox;

/**
 * \class PivotCalibrationViewPreferencePage
 * \brief Preferences page for the Pivot Calibration View plugin.
 * \ingroup uk_ac_ucl_cmic_igipivotcalibration_internal
 *
 */
class PivotCalibrationViewPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  /**
   * \brief Stores the name of the preferences node.
   */
  static const std::string PREFERENCES_NODE_NAME;

  /**
   * \brief Stores the name of the preference node that contains the name of the scale file.
   */
  static const std::string OUTPUT_DIRECTORY;

  PivotCalibrationViewPreferencePage();
  PivotCalibrationViewPreferencePage(const PivotCalibrationViewPreferencePage& other);
  ~PivotCalibrationViewPreferencePage();

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
  ctkPathLineEdit *m_OutputDirectoryChooser;

  bool            m_Initializing;

  berry::IPreferences::Pointer m_PivotCalibrationViewPreferencesNode;
};

#endif // PivotCalibrationViewPreferencePage_h


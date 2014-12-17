/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkNiftyViewApplicationPreferencePage_h
#define QmitkNiftyViewApplicationPreferencePage_h

#include <uk_ac_ucl_cmic_gui_qt_commonapps_Export.h>
#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>

class QWidget;
class QRadioButton;
class QDoubleSpinBox;
class QSpinBox;

/**
 * \class QmitkNiftyViewApplicationPreferencePage
 * \brief Preferences page for this plugin, providing application wide defaults.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftyview_internal
 *
 */
class CMIC_QT_COMMONAPPS QmitkNiftyViewApplicationPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  QmitkNiftyViewApplicationPreferencePage();
  QmitkNiftyViewApplicationPreferencePage(const QmitkNiftyViewApplicationPreferencePage& other);
  ~QmitkNiftyViewApplicationPreferencePage();

  static const std::string IMAGE_INITIALISATION_METHOD_NAME;
  static const std::string IMAGE_INITIALISATION_MIDAS;
  static const std::string IMAGE_INITIALISATION_LEVELWINDOW;
  static const std::string IMAGE_INITIALISATION_PERCENTAGE;
  static const std::string IMAGE_INITIALISATION_PERCENTAGE_NAME;
  static const std::string IMAGE_INITIALISATION_RANGE;
  static const std::string IMAGE_INITIALISATION_RANGE_LOWER_BOUND_NAME;
  static const std::string IMAGE_INITIALISATION_RANGE_UPPER_BOUND_NAME;

  void Init(berry::IWorkbench::Pointer workbench);

  void CreateQtControl(QWidget* widget);

  QWidget* GetQtControl() const;

  ///
  /// \see IPreferencePage::PerformOk()
  ///
  virtual bool PerformOk();

  ///
  /// \see IPreferencePage::PerformCancel()
  ///
  virtual void PerformCancel();

  ///
  /// \see IPreferencePage::Update()
  ///
  virtual void Update();

protected slots:

  void OnMIDASInitialisationRadioButtonChecked(bool);
  void OnLevelWindowRadioButtonChecked(bool);
  void OnImageDataRadioButtonChecked(bool);
  void OnIntensityRangeRadioButtonChecked(bool);

private:

  void UpdateSpinBoxes();

  QWidget        *m_MainControl;
  QRadioButton   *m_UseMidasInitialisationRadioButton;
  QRadioButton   *m_UseLevelWindowRadioButton;
  QRadioButton   *m_UseImageDataRadioButton;
  QDoubleSpinBox *m_PercentageOfDataRangeDoubleSpinBox;
  QRadioButton   *m_UseSetRange;
  QSpinBox       *m_RangeLowerBound;
  QSpinBox       *m_RangeUpperBound;
  bool m_Initializing;

  berry::IPreferences::Pointer m_PreferencesNode;
};

#endif /* QMITKNIFTYVIEWAPPLICATIONPREFERENCEPAGE_H_ */


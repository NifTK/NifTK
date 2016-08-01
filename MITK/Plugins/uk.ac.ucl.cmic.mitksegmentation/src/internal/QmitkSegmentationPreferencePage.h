/*===================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center, 
Division of Medical and Biological Informatics.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without 
even the implied warranty of MERCHANTABILITY or FITNESS FOR 
A PARTICULAR PURPOSE.

See LICENSE.txt or http://www.mitk.org for details.

===================================================================*/


#ifndef QmitkSegmentationPreferencePage_h
#define QmitkSegmentationPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>

class QWidget;
class QCheckBox;
class QRadioButton;
class QDoubleSpinBox;
class QPushButton;

class QmitkSegmentationPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  QmitkSegmentationPreferencePage();
  ~QmitkSegmentationPreferencePage();

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

  void OnVolumeRenderingCheckboxChecked(int);
  void OnSmoothingCheckboxChecked(int);
  void OnDefaultColourChanged();
  void OnResetDefaultColour();

protected:

  QWidget        *m_MainControl;
  QRadioButton   *m_RadioOutline;
  QRadioButton   *m_RadioOverlay;
  QCheckBox      *m_VolumeRenderingCheckBox;
  QCheckBox      *m_SmoothingCheckBox;
  QDoubleSpinBox *m_SmoothingSpinBox;
  QDoubleSpinBox *m_DecimationSpinBox;
  QDoubleSpinBox *m_ClosingSpinBox;
  QPushButton    *m_DefaultColorPushButton;
  QString         m_DefauleColorStyleSheet;
  std::string     m_DefaultColor;

  bool m_Initializing;

  berry::IPreferences::Pointer m_SegmentationPreferencesNode;
};

#endif /* QMITKDATAMANAGERPREFERENCEPAGE_H_ */

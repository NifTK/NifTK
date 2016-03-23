/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMorphologicalSegmentationViewPreferencePage_h
#define niftkMorphologicalSegmentationViewPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>

class QWidget;
class QPushButton;

/**
 * \class niftkMorphologicalSegmentationViewPreferencePage
 * \brief Preferences page for this plugin, enabling choice volume rendering on/off.
 * \ingroup uk_ac_ucl_cmic_midasmorphologicalsegmentor
 *
 */
class niftkMorphologicalSegmentationViewPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  /// \brief Stores the name of the preferences node.
  static const QString PREFERENCES_NODE_NAME;

  niftkMorphologicalSegmentationViewPreferencePage();
  niftkMorphologicalSegmentationViewPreferencePage(const niftkMorphologicalSegmentationViewPreferencePage& other);
  ~niftkMorphologicalSegmentationViewPreferencePage();

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

  void OnDefaultColourChanged();
  void OnResetDefaultColour();

protected:

  QWidget        *m_MainControl;
  QPushButton    *m_DefaultColorPushButton;
  QString         m_DefauleColorStyleSheet;
  QString     m_DefaultColor;

  bool m_Initializing;

  berry::IPreferences::Pointer m_niftkMorphologicalSegmentationViewPreferencesNode;
};

#endif

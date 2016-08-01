/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkVLStandardDisplayEditorPreferencePage_h
#define niftkVLStandardDisplayEditorPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>

class QWidget;
class QPushButton;

namespace niftk
{

/**
 * \class VLStandardDisplayEditorPreferencePage
 * \brief Preference page for VLStandardDisplayEditor, eg. setting the background colour.
 * \ingroup uk_ac_ucl_cmic_vlstandarddisplayeditor_internal
 */
struct VLStandardDisplayEditorPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:
  VLStandardDisplayEditorPreferencePage();

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

  /**
   * \brief Stores the name of the preference node that contains the background colour.
   */
  static const QString BACKGROUND_COLOR_PREFSKEY;
  static const unsigned int DEFAULT_BACKGROUND_COLOR;

public slots:

  void OnBackgroundColourClicked();

private:

  QWidget                      *m_MainControl;
  QPushButton                  *m_BackgroundColourButton;
  unsigned int                  m_BackgroundColour;
  berry::IPreferences::Pointer  m_VLStandardDisplayEditorPreferencesNode;
};

} // end namespace

#endif

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef SurfaceReconViewPreferencePage_h
#define SurfaceReconViewPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>
#include <QString>

class QWidget;
class QPushButton;

/**
 * \class SurfaceReconViewPreferencePage
 * \brief Preferences page for the Surface Reconstruction View plugin.
 * \ingroup uk_ac_ucl_cmic_igisurfacerecon_internal
 *
 */
class SurfaceReconViewPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  /// \brief Stores the name of the preferences node.
  static const std::string PREFERENCES_NODE_NAME;

  SurfaceReconViewPreferencePage();
  SurfaceReconViewPreferencePage(const SurfaceReconViewPreferencePage& other);
  ~SurfaceReconViewPreferencePage();

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

  berry::IPreferences::Pointer m_SurfaceReconViewPreferencesNode;
};

#endif // SurfaceReconViewPreferencePage_h


/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef SurfaceRegViewPreferencePage_h
#define SurfaceRegViewPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>
#include <QString>

class QWidget;
class QCheckBox;
class QSpinBox;

/**
 * \class SurfaceRegViewPreferencePage
 * \brief Preferences page for the Surface Based Registration View plugin.
 * \ingroup uk_ac_ucl_cmic_igisurfacereg_internal
 *
 */
class SurfaceRegViewPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  /// \brief Stores the name of the preferences node.
  static const std::string PREFERENCES_NODE_NAME;

  SurfaceRegViewPreferencePage();
  SurfaceRegViewPreferencePage(const SurfaceRegViewPreferencePage& other);
  ~SurfaceRegViewPreferencePage();

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
  QSpinBox       *m_MaximumIterations;
  QSpinBox       *m_MaximumPoints;
  QCheckBox      *m_TryDeformableRegistration;
  bool            m_Initializing;

  berry::IPreferences::Pointer m_SurfaceRegViewPreferencesNode;
};

#endif // SurfaceRegViewPreferencePage_h


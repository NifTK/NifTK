/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkUltrasoundReconstructionPreferencePage_h
#define niftkUltrasoundReconstructionPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>

class ctkDirectoryButton;

namespace niftk
{

/// \class UltrasoundReconstructionPreferencePage
/// \brief Preferences page for this plugin.
/// \ingroup uk_ac_ucl_cmic_ultrasoundreconstruction_internal
class UltrasoundReconstructionPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  UltrasoundReconstructionPreferencePage();
  UltrasoundReconstructionPreferencePage(const UltrasoundReconstructionPreferencePage& other);
  ~UltrasoundReconstructionPreferencePage();

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

protected:

  QWidget* m_MainControl;
  bool     m_Initializing;

  berry::IPreferences::Pointer m_UltrasoundReconstructionPreferencesNode;
};

}

#endif

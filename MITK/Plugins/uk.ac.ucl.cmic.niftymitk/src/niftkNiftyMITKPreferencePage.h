/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNiftyMITKPreferencePage_h
#define niftkNiftyMITKPreferencePage_h

#include <uk_ac_ucl_cmic_niftymitk_Export.h>

#include <niftkBaseApplicationPreferencePage.h>

class QWidget;

namespace niftk
{

/// \class NiftyMITKPreferencePage
/// \brief Preferences page for the NiftyMITK application, providing application wide defaults.
/// \ingroup uk_ac_ucl_cmic_niftyview_internal
class NIFTYMITK_EXPORT NiftyMITKPreferencePage : public BaseApplicationPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  NiftyMITKPreferencePage();
  NiftyMITKPreferencePage(const NiftyMITKPreferencePage& other);
  ~NiftyMITKPreferencePage();

  void Init(berry::IWorkbench::Pointer workbench) override;

  void CreateQtControl(QWidget* widget) override;

  /// \see IPreferencePage::PerformOk()
  virtual bool PerformOk() override;

  /// \see IPreferencePage::PerformCancel()
  virtual void PerformCancel() override;

  /// \see IPreferencePage::Update()
  virtual void Update() override;

};

}

#endif

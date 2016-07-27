/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNiftyViewPreferencePage_h
#define niftkNiftyViewPreferencePage_h

#include <uk_ac_ucl_cmic_niftyview_Export.h>

#include <niftkBaseApplicationPreferencePage.h>

class QWidget;

namespace niftk
{

/// \class NiftyViewPreferencePage
/// \brief Preferences page for the NiftyView application, providing application wide defaults.
/// \ingroup uk_ac_ucl_cmic_niftyview_internal
class NIFTYVIEW_EXPORT NiftyViewPreferencePage : public BaseApplicationPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  NiftyViewPreferencePage();
  NiftyViewPreferencePage(const NiftyViewPreferencePage& other);
  ~NiftyViewPreferencePage();

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

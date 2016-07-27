/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkSideViewerView_h
#define niftkSideViewerView_h

#include <uk_ac_ucl_cmic_sideviewer_Export.h>

#include <mitkRenderingManager.h>

#include <niftkBaseView.h>

namespace berry
{
class IBerryPreferences;
}

class QmitkRenderWindow;


namespace niftk
{
class SideViewerWidget;

/**
 * \class SideViewerView
 * \brief Base view component for MIDAS Segmentation widgets.
 *
 * \ingroup uk_ac_ucl_cmic_common
 *
 * \sa BaseView
 */
class SIDEVIEWER_EXPORT SideViewerView : public BaseView
{

  Q_OBJECT

public:

  SideViewerView();
  SideViewerView(const SideViewerView& other);
  virtual ~SideViewerView();

protected:

  /// \brief Creates the GUI parts.
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief Called by framework, sets the focus on a specific widget.
  virtual void SetFocus();

  /// \brief Decorates a DataNode according to the user preference settings, or requirements for binary images.
  virtual void ApplyDisplayOptions(mitk::DataNode* node);

  /// \brief Called when preferences are updated.
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*);

  /// \brief Retrieve's the pref values from preference service, and store locally.
  virtual void RetrievePreferenceValues();

  /// \brief Derived classes decide which preferences are actually read.
  virtual QString GetPreferencesNodeName();

private:

  /// \brief Rendering manager of the internal viewer.
  /// This class holds a smart pointer so that it does not get destroyed too early.
  mitk::RenderingManager::Pointer m_RenderingManager;

  /// \brief Provides an additional view of the segmented image, so plugin can be used on second monitor.
  SideViewerWidget *m_SideViewerWidget;
};

}

#endif

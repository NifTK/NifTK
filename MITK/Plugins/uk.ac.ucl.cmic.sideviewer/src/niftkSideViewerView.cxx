/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSideViewerView.h"

#include <QTimer>

#include <berryIBerryPreferences.h>
#include <berryPlatform.h>

#include <mitkVtkResliceInterpolationProperty.h>

#include <niftkSingleViewerWidget.h>

#include "niftkSideViewerWidget.h"

#include "internal/niftkPluginActivator.h"


namespace niftk
{

class SideViewerViewPrivate
{
public:
  SideViewerViewPrivate(SideViewerView* q);
  ~SideViewerViewPrivate();

  static bool s_OptionsProcessed;

  static bool AreOptionsProcessed();

  void ProcessOptions();

  void ProcessDisplayConventionOption();

  /// \brief Rendering manager of the internal viewer.
  /// This class holds a smart pointer so that it does not get destroyed too early.
  mitk::RenderingManager::Pointer m_RenderingManager;

  /// \brief Provides an additional view of the segmented image, so plugin can be used on second monitor.
  SideViewerWidget *m_SideViewerWidget;

  SideViewerView* q_ptr;
};

bool SideViewerViewPrivate::s_OptionsProcessed = false;

//-----------------------------------------------------------------------------
SideViewerViewPrivate::SideViewerViewPrivate(SideViewerView* q)
: q_ptr(q),
  m_RenderingManager(nullptr),
  m_SideViewerWidget(nullptr)
{
}


//-----------------------------------------------------------------------------
SideViewerViewPrivate::~SideViewerViewPrivate()
{
}


//-----------------------------------------------------------------------------
bool SideViewerViewPrivate::AreOptionsProcessed()
{
  return s_OptionsProcessed;
}


//-----------------------------------------------------------------------------
void SideViewerViewPrivate::ProcessOptions()
{
  this->ProcessDisplayConventionOption();

  s_OptionsProcessed = true;
}


// --------------------------------------------------------------------------
void SideViewerViewPrivate::ProcessDisplayConventionOption()
{
  ctkPluginContext* pluginContext = PluginActivator::GetInstance()->GetContext();

  QString displayConventionArg = pluginContext->getProperty("applicationArgs.display-convention").toString();

  if (!displayConventionArg.isNull())
  {
    int displayConvention;

    if (displayConventionArg == "radio")
    {
      displayConvention = DISPLAY_CONVENTION_RADIO;
    }
    else if (displayConventionArg == "neuro")
    {
      displayConvention = DISPLAY_CONVENTION_NEURO;
    }
    else if (displayConventionArg == "radio-x-flipped")
    {
      displayConvention = DISPLAY_CONVENTION_RADIO_X_FLIPPED;
    }
    else
    {
      MITK_ERROR << "Invalid display convention: " << displayConventionArg.toStdString();
      MITK_ERROR << "Supported conventions are: 'radio', 'neuro' and 'radio-x-flipped'.";
      return;
    }

    m_SideViewerWidget->GetViewer()->SetDisplayConvention(displayConvention);
  }

}


//-----------------------------------------------------------------------------
SideViewerView::SideViewerView()
: d(new SideViewerViewPrivate(this))
{
}


//-----------------------------------------------------------------------------
SideViewerView::SideViewerView(
    const SideViewerView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
SideViewerView::~SideViewerView()
{
  if (d->m_SideViewerWidget)
  {
    delete d->m_SideViewerWidget;
  }
}


//-----------------------------------------------------------------------------
void SideViewerView::CreateQtPartControl(QWidget *parent)
{
  if (!d->m_SideViewerWidget)
  {
    d->m_RenderingManager = mitk::RenderingManager::New();
    d->m_RenderingManager->SetDataStorage(this->GetDataStorage());

    d->m_SideViewerWidget = new SideViewerWidget(this, parent, d->m_RenderingManager);
    d->m_SideViewerWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    // Retrieving preferences done in another method so we can call it on startup, and when prefs change.
    this->RetrievePreferenceValues();

    /// The command line arguments should be processed after the widget has been created
    /// and it becomes visible.
    ///
    /// Here we are in the function that creates the widget, that means, the widget will have
    /// been created right after this function returns. So that we do not need to deal with
    /// event filters and such, we delay the call to process the command line arguments by
    /// one millisecond. This leaves time for this function to return, and the arguments will
    /// be processed as soon as possible.
    if (!SideViewerViewPrivate::AreOptionsProcessed())
    {
      QTimer::singleShot(1, this, SLOT(ProcessOptions()));
    }
  }
}


//-----------------------------------------------------------------------------
void SideViewerView::ProcessOptions()
{
  d->ProcessOptions();
}


//-----------------------------------------------------------------------------
void SideViewerView::SetFocus()
{
  d->m_SideViewerWidget->SetFocused();
}


//-----------------------------------------------------------------------------
void SideViewerView::ApplyDisplayOptions(mitk::DataNode* node)
{
  if (!node)
  {
    return;
  }

  bool isBinary = false;
  if (node->GetBoolProperty("binary", isBinary) && isBinary)
  {
    node->ReplaceProperty("reslice interpolation", mitk::VtkResliceInterpolationProperty::New(VTK_RESLICE_NEAREST), const_cast<const mitk::BaseRenderer*>((mitk::BaseRenderer*)NULL));
    node->SetBoolProperty("outline binary", true);
    node->SetFloatProperty ("outline width", 1.0);
    node->SetBoolProperty("showVolume", false);
    node->SetBoolProperty("volumerendering", false);
    node->SetOpacity(1.0);
  }
}


//-----------------------------------------------------------------------------
void SideViewerView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void SideViewerView::RetrievePreferenceValues()
{
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();

  assert( prefService );

  berry::IPreferences::Pointer prefs =
      prefService->GetSystemPreferences()->Node(this->GetPreferencesNodeName());

  assert( prefs );

  // ...
}


//-----------------------------------------------------------------------------
QString SideViewerView::GetPreferencesNodeName()
{
  return "/uk_ac_ucl_cmic_sideviewer";
}

}

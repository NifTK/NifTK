/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkVLStandardDisplayEditor.h"

#include <berryUIException.h>
#include <berryIWorkbenchPage.h>
#include <berryIPreferencesService.h>
#include <berryIPartListener.h>

// Note:
// This header must be included before mitkOclResourceService.h to avoid name clash between Xlib.h and Qt.
// Both headers define a 'None' constant. The header below undefines it to avoid compile error with gcc.
#include <niftkVLStandardDisplayWidget.h>

#include <mitkColorProperty.h>
#include <mitkDataStorageEditorInput.h>
#include <mitkIDataStorageService.h>
#include <mitkOclResourceService.h>

#include <internal/niftkVLStandardDisplayEditorPreferencePage.h>
#include <internal/niftkVLStandardDisplayEditorActivator.h>

namespace niftk
{

const char* VLStandardDisplayEditor::EDITOR_ID = "org.mitk.editors.vlstandarddisplayeditor";

/**
 * \class VLStandardDisplayEditorPrivate
 * \brief PIMPL pattern implementation of VLStandardDisplayEditor.
 */
class VLStandardDisplayEditorPrivate
{
public:

  VLStandardDisplayEditorPrivate();
  ~VLStandardDisplayEditorPrivate();

  niftk::VLStandardDisplayWidget* m_VLStandardDisplayWidget;
  std::string m_FirstBackgroundColor;
  std::string m_SecondBackgroundColor;
  QScopedPointer<berry::IPartListener> m_PartListener;
};


/**
 * \class IGIOverlayWidgetPartListener
 * \brief Used to handle interaction with the contained overlay
 * editor widget when this VLStandardDisplayEditor is opened/closed etc.
 */
struct IGIOverlayWidgetPartListener : public berry::IPartListener
{
  berryObjectMacro(IGIOverlayWidgetPartListener)

  //---------------------------------------------------------------------------
  IGIOverlayWidgetPartListener(VLStandardDisplayEditorPrivate* dd)
    : d(dd)
  {}

  //---------------------------------------------------------------------------
  Events::Types GetPartEventTypes() const override
  {
    return Events::CLOSED | Events::HIDDEN | Events::VISIBLE;
  }

  //---------------------------------------------------------------------------
  void PartClosed(const berry::IWorkbenchPartReference::Pointer& partRef) override
  {
    if (partRef->GetId() == VLStandardDisplayEditor::EDITOR_ID)
    {
      VLStandardDisplayEditor::Pointer editor = partRef->GetPart(false).Cast<VLStandardDisplayEditor>();
      if (d->m_VLStandardDisplayWidget == editor->GetVLStandardDisplayWidget())
      {
        // Call editor to turn things off as the widget is being closed.
      }
    }
  }

  //---------------------------------------------------------------------------
  void PartHidden (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == VLStandardDisplayEditor::EDITOR_ID)
    {
      VLStandardDisplayEditor::Pointer editor = partRef->GetPart(false).Cast<VLStandardDisplayEditor>();
      if (d->m_VLStandardDisplayWidget == editor->GetVLStandardDisplayWidget())
      {
        // Call editor to turn things off as the widget is being hidden.
      }
    }
  }

  //---------------------------------------------------------------------------
  void PartVisible (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == VLStandardDisplayEditor::EDITOR_ID)
    {
      VLStandardDisplayEditor::Pointer editor = partRef->GetPart(false).Cast<VLStandardDisplayEditor>();
      if (d->m_VLStandardDisplayWidget == editor->GetVLStandardDisplayWidget())
      {
        // Call editor to turn things on as the widget is being made visible.
      }
    }
  }

private:

  VLStandardDisplayEditorPrivate* const d;

};


//-----------------------------------------------------------------------------
VLStandardDisplayEditorPrivate::VLStandardDisplayEditorPrivate()
  : m_VLStandardDisplayWidget(0)
  , m_PartListener(new IGIOverlayWidgetPartListener(this))
{}


//-----------------------------------------------------------------------------
VLStandardDisplayEditorPrivate::~VLStandardDisplayEditorPrivate()
{
}

//-----------------------------------------------------------------------------
VLStandardDisplayEditor::VLStandardDisplayEditor()
  : d(new VLStandardDisplayEditorPrivate)
{
}


//-----------------------------------------------------------------------------
VLStandardDisplayEditor::~VLStandardDisplayEditor()
{
  this->disconnect();
  this->GetSite()->GetPage()->RemovePartListener(d->m_PartListener.data());
}


//-----------------------------------------------------------------------------
niftk::VLStandardDisplayWidget* VLStandardDisplayEditor::GetVLStandardDisplayWidget()
{
  return d->m_VLStandardDisplayWidget;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *VLStandardDisplayEditor::GetActiveQmitkRenderWindow() const
{
  return nullptr;
}


//-----------------------------------------------------------------------------
QHash<QString, QmitkRenderWindow *> VLStandardDisplayEditor::GetQmitkRenderWindows() const
{
  return QHash<QString, QmitkRenderWindow *>();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *VLStandardDisplayEditor::GetQmitkRenderWindow(const QString &id) const
{
  return nullptr;
}


//-----------------------------------------------------------------------------
mitk::Point3D VLStandardDisplayEditor::GetSelectedPosition(const QString & id) const
{
  // Not implemented.
  mitk::Point3D point;
  point[0] = 0;
  point[1] = 0;
  point[2] = 0;
  return point;
}


//-----------------------------------------------------------------------------
void VLStandardDisplayEditor::SetSelectedPosition(const mitk::Point3D &pos, const QString &id)
{
  // Not implemented.
}


//-----------------------------------------------------------------------------
void VLStandardDisplayEditor::EnableDecorations(bool /*enable*/, const QStringList & /*decorations*/)
{
}


//-----------------------------------------------------------------------------
bool VLStandardDisplayEditor::IsDecorationEnabled(const QString & /*decoration*/) const
{
  return false;
}


//-----------------------------------------------------------------------------
QStringList VLStandardDisplayEditor::GetDecorations() const
{
  QStringList decorations;
  return decorations;
}


//-----------------------------------------------------------------------------
mitk::SlicesRotator* VLStandardDisplayEditor::GetSlicesRotator() const
{
  return nullptr;
}


//-----------------------------------------------------------------------------
mitk::SlicesSwiveller* VLStandardDisplayEditor::GetSlicesSwiveller() const
{
  return nullptr;
}


//-----------------------------------------------------------------------------
void VLStandardDisplayEditor::EnableSlicingPlanes(bool /*enable*/)
{
}


//-----------------------------------------------------------------------------
bool VLStandardDisplayEditor::IsSlicingPlanesEnabled() const
{
  return false;
}


//-----------------------------------------------------------------------------
void VLStandardDisplayEditor::EnableLinkedNavigation(bool /*enable*/)
{
}


//-----------------------------------------------------------------------------
bool VLStandardDisplayEditor::IsLinkedNavigationEnabled() const
{
  return false;
}


//-----------------------------------------------------------------------------
void VLStandardDisplayEditor::CreateQtPartControl(QWidget* parent)
{
  if (d->m_VLStandardDisplayWidget == 0)
  {

    mitk::DataStorage::Pointer ds = this->GetDataStorage();

    d->m_VLStandardDisplayWidget = new niftk::VLStandardDisplayWidget(parent);
    d->m_VLStandardDisplayWidget->SetDataStorage(ds);

    QHBoxLayout* layout = new QHBoxLayout(parent);
    layout->setContentsMargins(0,0,0,0);
    layout->addWidget(d->m_VLStandardDisplayWidget);

    ctkPluginContext*     context     = niftk::VLStandardDisplayEditorActivator::/*GetDefault()->*/getContext();
    ctkServiceReference   serviceRef  = context->getServiceReference<OclResourceService>();
    OclResourceService*   oclService  = context->getService<OclResourceService>(serviceRef);
    if (oclService == NULL)
    {
      mitkThrow() << "Failed to find OpenCL resource service." << std::endl;
    }
    d->m_VLStandardDisplayWidget->SetOclResourceService(oclService);

    this->GetSite()->GetPage()->AddPartListener(d->m_PartListener.data());

    QMetaObject::invokeMethod(this, "OnPreferencesChanged", Qt::QueuedConnection);

    this->RequestUpdate();
  }
}


//-----------------------------------------------------------------------------
void VLStandardDisplayEditor::OnPreferencesChanged()
{
  this->OnPreferencesChanged(dynamic_cast<berry::IBerryPreferences*>(this->GetPreferences().GetPointer()));
}


//-----------------------------------------------------------------------------
void VLStandardDisplayEditor::OnPreferencesChanged(const berry::IBerryPreferences* prefs)
{
  // 0xAABBGGRR
  unsigned int   backgroundColour = prefs->GetInt(VLStandardDisplayEditorPreferencePage::BACKGROUND_COLOR_PREFSKEY,
                                                  VLStandardDisplayEditorPreferencePage::DEFAULT_BACKGROUND_COLOR);
  if (d->m_VLStandardDisplayWidget != 0)
  {
    d->m_VLStandardDisplayWidget->SetBackgroundColour(backgroundColour);
  }
}


//-----------------------------------------------------------------------------
void VLStandardDisplayEditor::SetFocus()
{
  if (d->m_VLStandardDisplayWidget != 0)
  {
    d->m_VLStandardDisplayWidget->setFocus();
  }
}

} // end namespace

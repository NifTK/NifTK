/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIVLVideoOverlayEditor.h"

#include <berryUIException.h>
#include <berryIWorkbenchPage.h>
#include <berryIPreferencesService.h>
#include <berryIPartListener.h>

// Note:
// This header must be included before mitkOclResourceService.h to avoid name clash between Xlib.h and Qt.
// Both headers define a 'None' constant. The header below undefines it to avoid compile error with gcc.
#include <niftkVLVideoOverlayWidget.h>

#include <mitkColorProperty.h>
#include <mitkDataStorageEditorInput.h>
#include <mitkIDataStorageService.h>
#include <mitkOclResourceService.h>

#include <internal/niftkIGIVLVideoOverlayEditorPreferencePage.h>
#include <internal/niftkIGIVLVideoOverlayEditorActivator.h>

namespace niftk
{

const char* IGIVLVideoOverlayEditor::EDITOR_ID = "org.mitk.editors.IGIVLVideoOverlayeditor";

/**
 * \class IGIVLVideoOverlayEditorPrivate
 * \brief PIMPL pattern implementation of IGIVLVideoOverlayEditor.
 */
class IGIVLVideoOverlayEditorPrivate
{
public:

  IGIVLVideoOverlayEditorPrivate();
  ~IGIVLVideoOverlayEditorPrivate();

  niftk::VLVideoOverlayWidget* m_VLVideoOverlayWidget;
  std::string m_FirstBackgroundColor;
  std::string m_SecondBackgroundColor;
  QScopedPointer<berry::IPartListener> m_PartListener;
};


/**
 * \class IGIOverlayWidgetPartListener
 * \brief Used to handle interaction with the contained overlay
 * editor widget when this IGIVLVideoOverlayEditor is opened/closed etc.
 */
struct IGIOverlayWidgetPartListener : public berry::IPartListener
{
  berryObjectMacro(IGIOverlayWidgetPartListener)

  //---------------------------------------------------------------------------
  IGIOverlayWidgetPartListener(IGIVLVideoOverlayEditorPrivate* dd)
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
    if (partRef->GetId() == IGIVLVideoOverlayEditor::EDITOR_ID)
    {
      IGIVLVideoOverlayEditor::Pointer editor = partRef->GetPart(false).Cast<IGIVLVideoOverlayEditor>();
      if (d->m_VLVideoOverlayWidget == editor->GetVLVideoOverlayWidget())
      {
        // Call editor to turn things off as the widget is being closed.
      }
    }
  }

  //---------------------------------------------------------------------------
  void PartHidden (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == IGIVLVideoOverlayEditor::EDITOR_ID)
    {
      IGIVLVideoOverlayEditor::Pointer editor = partRef->GetPart(false).Cast<IGIVLVideoOverlayEditor>();
      if (d->m_VLVideoOverlayWidget == editor->GetVLVideoOverlayWidget())
      {
        // Call editor to turn things off as the widget is being hidden.
      }
    }
  }

  //---------------------------------------------------------------------------
  void PartVisible (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == IGIVLVideoOverlayEditor::EDITOR_ID)
    {
      IGIVLVideoOverlayEditor::Pointer editor = partRef->GetPart(false).Cast<IGIVLVideoOverlayEditor>();
      if (d->m_VLVideoOverlayWidget == editor->GetVLVideoOverlayWidget())
      {
        // Call editor to turn things on as the widget is being made visible.
      }
    }
  }

private:

  IGIVLVideoOverlayEditorPrivate* const d;

};


//-----------------------------------------------------------------------------
IGIVLVideoOverlayEditorPrivate::IGIVLVideoOverlayEditorPrivate()
  : m_VLVideoOverlayWidget(0)
  , m_PartListener(new IGIOverlayWidgetPartListener(this))
{}


//-----------------------------------------------------------------------------
IGIVLVideoOverlayEditorPrivate::~IGIVLVideoOverlayEditorPrivate()
{
}

//-----------------------------------------------------------------------------
IGIVLVideoOverlayEditor::IGIVLVideoOverlayEditor()
  : d(new IGIVLVideoOverlayEditorPrivate)
{
}


//-----------------------------------------------------------------------------
IGIVLVideoOverlayEditor::~IGIVLVideoOverlayEditor()
{
  this->disconnect();
  this->GetSite()->GetPage()->RemovePartListener(d->m_PartListener.data());
}


//-----------------------------------------------------------------------------
niftk::VLVideoOverlayWidget* IGIVLVideoOverlayEditor::GetVLVideoOverlayWidget()
{
  return d->m_VLVideoOverlayWidget;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *IGIVLVideoOverlayEditor::GetActiveQmitkRenderWindow() const
{
  return nullptr;
}


//-----------------------------------------------------------------------------
QHash<QString, QmitkRenderWindow *> IGIVLVideoOverlayEditor::GetQmitkRenderWindows() const
{
  return QHash<QString, QmitkRenderWindow *>();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *IGIVLVideoOverlayEditor::GetQmitkRenderWindow(const QString &id) const
{
  return nullptr;
}


//-----------------------------------------------------------------------------
mitk::Point3D IGIVLVideoOverlayEditor::GetSelectedPosition(const QString & id) const
{
  // Not implemented.
  mitk::Point3D point;
  point[0] = 0;
  point[1] = 0;
  point[2] = 0;
  return point;
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayEditor::SetSelectedPosition(const mitk::Point3D &pos, const QString &id)
{
  // Not implemented.
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayEditor::EnableDecorations(bool /*enable*/, const QStringList & /*decorations*/)
{
}


//-----------------------------------------------------------------------------
bool IGIVLVideoOverlayEditor::IsDecorationEnabled(const QString & /*decoration*/) const
{
  return false;
}


//-----------------------------------------------------------------------------
QStringList IGIVLVideoOverlayEditor::GetDecorations() const
{
  QStringList decorations;
  return decorations;
}


//-----------------------------------------------------------------------------
mitk::SlicesRotator* IGIVLVideoOverlayEditor::GetSlicesRotator() const
{
  return nullptr;
}


//-----------------------------------------------------------------------------
mitk::SlicesSwiveller* IGIVLVideoOverlayEditor::GetSlicesSwiveller() const
{
  return nullptr;
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayEditor::EnableSlicingPlanes(bool /*enable*/)
{
}


//-----------------------------------------------------------------------------
bool IGIVLVideoOverlayEditor::IsSlicingPlanesEnabled() const
{
  return false;
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayEditor::EnableLinkedNavigation(bool /*enable*/)
{
}


//-----------------------------------------------------------------------------
bool IGIVLVideoOverlayEditor::IsLinkedNavigationEnabled() const
{
  return false;
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayEditor::CreateQtPartControl(QWidget* parent)
{
  if (d->m_VLVideoOverlayWidget == 0)
  {

    mitk::DataStorage::Pointer ds = this->GetDataStorage();

    d->m_VLVideoOverlayWidget = new niftk::VLVideoOverlayWidget(parent);
    d->m_VLVideoOverlayWidget->SetDataStorage(ds);

    QHBoxLayout* layout = new QHBoxLayout(parent);
    layout->setContentsMargins(0,0,0,0);
    layout->addWidget(d->m_VLVideoOverlayWidget);

    ctkPluginContext*     context     = niftk::IGIVLVideoOverlayEditorActivator::/*GetDefault()->*/getContext();
    ctkServiceReference   serviceRef  = context->getServiceReference<OclResourceService>();
    OclResourceService*   oclService  = context->getService<OclResourceService>(serviceRef);
    if (oclService == NULL)
    {
      mitkThrow() << "Failed to find OpenCL resource service." << std::endl;
    }
    d->m_VLVideoOverlayWidget->SetOclResourceService(oclService);

    this->GetSite()->GetPage()->AddPartListener(d->m_PartListener.data());

    QMetaObject::invokeMethod(this, "OnPreferencesChanged", Qt::QueuedConnection);

    this->RequestUpdate();
  }
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayEditor::OnPreferencesChanged()
{
  this->OnPreferencesChanged(dynamic_cast<berry::IBerryPreferences*>(this->GetPreferences().GetPointer()));
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayEditor::OnPreferencesChanged(const berry::IBerryPreferences* prefs)
{
  // 0xAABBGGRR
  unsigned int   backgroundColour = prefs->GetInt(IGIVLVideoOverlayEditorPreferencePage::BACKGROUND_COLOR_PREFSKEY,
                                                  IGIVLVideoOverlayEditorPreferencePage::DEFAULT_BACKGROUND_COLOR);
  std::string calibrationFileName = prefs->Get(IGIVLVideoOverlayEditorPreferencePage::CALIBRATION_FILE_NAME, "").toStdString();

  if (d->m_VLVideoOverlayWidget != 0)
  {
    d->m_VLVideoOverlayWidget->SetBackgroundColour(backgroundColour);
    d->m_VLVideoOverlayWidget->SetEyeHandFileName(calibrationFileName);
  }
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayEditor::SetFocus()
{
  if (d->m_VLVideoOverlayWidget != 0)
  {
    d->m_VLVideoOverlayWidget->setFocus();
  }
}

} // end namespace

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIUltrasoundOverlayEditor.h"

#include <berryUIException.h>
#include <berryIWorkbenchPage.h>
#include <berryIPreferencesService.h>
#include <berryIPartListener.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>

#include <QWidget>
#include <QDateTime>
#include <QFile>
#include <QTextStream>
#include <QDir>

#include <mitkColorProperty.h>
#include <mitkGlobalInteraction.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateProperty.h>

#include <mitkDataStorageEditorInput.h>
#include <mitkIDataStorageService.h>

#include "niftkIGIUltrasoundOverlayWidget.h"
#include <internal/niftkIGIUltrasoundOverlayEditorPreferencePage.h>
#include <internal/niftkIGIUltrasoundOverlayEditorActivator.h>

namespace niftk
{

const char* IGIUltrasoundOverlayEditor::EDITOR_ID = "org.mitk.editors.igiultrasoundoverlayeditor";

/**
 * \class IGIUltrasoundOverlayEditorPrivate
 * \brief PIMPL pattern implementation of IGIUltrasoundOverlayEditor.
 */
class IGIUltrasoundOverlayEditorPrivate
{
public:

  IGIUltrasoundOverlayEditorPrivate();
  ~IGIUltrasoundOverlayEditorPrivate();

  niftk::IGIUltrasoundOverlayWidget* m_IGIUltrasoundOverlayWidget;
  std::string m_FirstBackgroundColor;
  std::string m_SecondBackgroundColor;
  QScopedPointer<berry::IPartListener> m_PartListener;
};


/**
 * \class IGIOverlayWidgetPartListener
 * \brief Used to handle interaction with the contained overlay
 * editor widget when this IGIUltrasoundOverlayEditor is opened/closed etc.
 */
struct IGIOverlayWidgetPartListener : public berry::IPartListener
{
  berryObjectMacro(IGIOverlayWidgetPartListener)

  //---------------------------------------------------------------------------
  IGIOverlayWidgetPartListener(IGIUltrasoundOverlayEditorPrivate* dd)
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
    if (partRef->GetId() == IGIUltrasoundOverlayEditor::EDITOR_ID)
    {
      IGIUltrasoundOverlayEditor::Pointer editor = partRef->GetPart(false).Cast<IGIUltrasoundOverlayEditor>();
      if (d->m_IGIUltrasoundOverlayWidget == editor->GetIGIUltrasoundOverlayWidget())
      {
        // Call editor to turn things off as the widget is being closed.
      }
    }
  }

  //---------------------------------------------------------------------------
  void PartHidden (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == IGIUltrasoundOverlayEditor::EDITOR_ID)
    {
      IGIUltrasoundOverlayEditor::Pointer editor = partRef->GetPart(false).Cast<IGIUltrasoundOverlayEditor>();
      if (d->m_IGIUltrasoundOverlayWidget == editor->GetIGIUltrasoundOverlayWidget())
      {
        // Call editor to turn things off as the widget is being hidden.
      }
    }
  }

  //---------------------------------------------------------------------------
  void PartVisible (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == IGIUltrasoundOverlayEditor::EDITOR_ID)
    {
      IGIUltrasoundOverlayEditor::Pointer editor = partRef->GetPart(false).Cast<IGIUltrasoundOverlayEditor>();
      if (d->m_IGIUltrasoundOverlayWidget == editor->GetIGIUltrasoundOverlayWidget())
      {
        // Call editor to turn things on as the widget is being made visible.
      }
    }
  }

private:

  IGIUltrasoundOverlayEditorPrivate* const d;

};


//-----------------------------------------------------------------------------
IGIUltrasoundOverlayEditorPrivate::IGIUltrasoundOverlayEditorPrivate()
  : m_IGIUltrasoundOverlayWidget(0)
  , m_PartListener(new IGIOverlayWidgetPartListener(this))
{}


//-----------------------------------------------------------------------------
IGIUltrasoundOverlayEditorPrivate::~IGIUltrasoundOverlayEditorPrivate()
{
}

//-----------------------------------------------------------------------------
IGIUltrasoundOverlayEditor::IGIUltrasoundOverlayEditor()
  : d(new IGIUltrasoundOverlayEditorPrivate)
{
}


//-----------------------------------------------------------------------------
IGIUltrasoundOverlayEditor::~IGIUltrasoundOverlayEditor()
{
  this->disconnect();
  this->GetSite()->GetPage()->RemovePartListener(d->m_PartListener.data());
}


//-----------------------------------------------------------------------------
niftk::IGIUltrasoundOverlayWidget* IGIUltrasoundOverlayEditor::GetIGIUltrasoundOverlayWidget()
{
  return d->m_IGIUltrasoundOverlayWidget;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *IGIUltrasoundOverlayEditor::GetActiveQmitkRenderWindow() const
{
  return d->m_IGIUltrasoundOverlayWidget->GetActiveQmitkRenderWindow();
}


//-----------------------------------------------------------------------------
QHash<QString, QmitkRenderWindow *> IGIUltrasoundOverlayEditor::GetQmitkRenderWindows() const
{
  return d->m_IGIUltrasoundOverlayWidget->GetQmitkRenderWindows();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *IGIUltrasoundOverlayEditor::GetQmitkRenderWindow(const QString &id) const
{
  return d->m_IGIUltrasoundOverlayWidget->GetQmitkRenderWindow(id);
}


//-----------------------------------------------------------------------------
mitk::Point3D IGIUltrasoundOverlayEditor::GetSelectedPosition(const QString & id) const
{
  // Not implemented.
  mitk::Point3D point;
  point[0] = 0;
  point[1] = 0;
  point[2] = 0;
  return point;
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayEditor::SetSelectedPosition(const mitk::Point3D &pos, const QString &id)
{
  // Not implemented.
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayEditor::EnableDecorations(bool /*enable*/, const QStringList & /*decorations*/)
{
}


//-----------------------------------------------------------------------------
bool IGIUltrasoundOverlayEditor::IsDecorationEnabled(const QString & /*decoration*/) const
{
  return false;
}


//-----------------------------------------------------------------------------
QStringList IGIUltrasoundOverlayEditor::GetDecorations() const
{
  QStringList decorations;
  return decorations;
}


//-----------------------------------------------------------------------------
mitk::SlicesRotator* IGIUltrasoundOverlayEditor::GetSlicesRotator() const
{
  return NULL;
}


//-----------------------------------------------------------------------------
mitk::SlicesSwiveller* IGIUltrasoundOverlayEditor::GetSlicesSwiveller() const
{
  return NULL;
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayEditor::EnableSlicingPlanes(bool /*enable*/)
{
}


//-----------------------------------------------------------------------------
bool IGIUltrasoundOverlayEditor::IsSlicingPlanesEnabled() const
{
  return false;
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayEditor::EnableLinkedNavigation(bool /*enable*/)
{
}


//-----------------------------------------------------------------------------
bool IGIUltrasoundOverlayEditor::IsLinkedNavigationEnabled() const
{
  return false;
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayEditor::CreateQtPartControl(QWidget* parent)
{
  if (d->m_IGIUltrasoundOverlayWidget == 0)
  {
    mitk::DataStorage::Pointer ds = this->GetDataStorage();

    d->m_IGIUltrasoundOverlayWidget = new niftk::IGIUltrasoundOverlayWidget(parent);
    d->m_IGIUltrasoundOverlayWidget->SetDataStorage(ds);

    QHBoxLayout* layout = new QHBoxLayout(parent);
    layout->setContentsMargins(0,0,0,0);
    layout->addWidget(d->m_IGIUltrasoundOverlayWidget);

    this->GetSite()->GetPage()->AddPartListener(d->m_PartListener.data());

    QMetaObject::invokeMethod(this, "OnPreferencesChanged", Qt::QueuedConnection);

    this->RequestUpdate();

    // Finally: Listen to update pulse coming off of event bus. This pulse comes from the data manager updating.
    ctkServiceReference ref = niftk::IGIUltrasoundOverlayEditorActivator::getContext()->getServiceReference<ctkEventAdmin>();
    if (ref)
    {
      ctkEventAdmin* eventAdmin = niftk::IGIUltrasoundOverlayEditorActivator::getContext()->getService<ctkEventAdmin>(ref);
      
      ctkDictionary propertiesIGI;
      propertiesIGI[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
      eventAdmin->subscribeSlot(this, SLOT(OnIGIUpdate(ctkEvent)), propertiesIGI);      
    }
  }
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayEditor::OnPreferencesChanged()
{
  this->OnPreferencesChanged(dynamic_cast<berry::IBerryPreferences*>(this->GetPreferences().GetPointer()));
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayEditor::OnPreferencesChanged(const berry::IBerryPreferences* prefs)
{
  // Enable change of logo. If no DepartmentLogo was set explicitly, MBILogo is used.
  // Set new department logo by prefs->Set("DepartmentLogo", "PathToImage");

  foreach (QString key, prefs->Keys())
  {
    if( key == "DepartmentLogo")
    {
      QString departmentLogoLocation = prefs->Get("DepartmentLogo", "");

      if (departmentLogoLocation.isEmpty())
      {
        d->m_IGIUltrasoundOverlayWidget->DisableDepartmentLogo();
      }
      else
      {
        d->m_IGIUltrasoundOverlayWidget->SetDepartmentLogoPath(departmentLogoLocation);
        d->m_IGIUltrasoundOverlayWidget->EnableDepartmentLogo();
      }
      break;
    }
  }
 
  // Preferences for gradient background
  float color = 255.0;
  QString firstColorName = prefs->Get(IGIUltrasoundOverlayEditorPreferencePage::FIRST_BACKGROUND_COLOUR, "");
  QColor firstColor(firstColorName);
  mitk::Color upper;
  if (firstColorName=="") // default values
  {
    upper[0] = 0;
    upper[1] = 0;
    upper[2] = 0;
  }
  else
  {
    upper[0] = firstColor.red() / color;
    upper[1] = firstColor.green() / color;
    upper[2] = firstColor.blue() / color;
  }

  QString secondColorName = prefs->Get(IGIUltrasoundOverlayEditorPreferencePage::SECOND_BACKGROUND_COLOUR, "");
  QColor secondColor(secondColorName);
  mitk::Color lower;
  if (secondColorName=="") // default values
  {
    lower[0] = 0;
    lower[1] = 0;
    lower[2] = 0;
  }
  else
  {
    lower[0] = secondColor.red() / color;
    lower[1] = secondColor.green() / color;
    lower[2] = secondColor.blue() / color;
  }
  d->m_IGIUltrasoundOverlayWidget->SetGradientBackgroundColors(upper, lower);
  d->m_IGIUltrasoundOverlayWidget->EnableGradientBackground();
  d->m_IGIUltrasoundOverlayWidget->SetClipToImagePlane(prefs->GetBool(IGIUltrasoundOverlayEditorPreferencePage::CLIP_TO_IMAGE_PLANE, true));
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayEditor::SetFocus()
{
  if (d->m_IGIUltrasoundOverlayWidget != 0)
  {
    d->m_IGIUltrasoundOverlayWidget->setFocus();
  }
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayEditor::OnIGIUpdate(const ctkEvent& event)
{
  d->m_IGIUltrasoundOverlayWidget->Update();
}

} // end namespace

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "IGIVideoOverlayEditor.h"

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

#include <niftkIGIVideoOverlayWidget.h>
#include <internal/IGIVideoOverlayEditorPreferencePage.h>
#include <internal/IGIVideoOverlayEditorActivator.h>

const char* IGIVideoOverlayEditor::EDITOR_ID = "org.mitk.editors.IGIVideoOverlayEditor";

/**
 * \class IGIVideoOverlayEditorPrivate
 * \brief PIMPL pattern implementation of IGIVideoOverlayEditor.
 */
class IGIVideoOverlayEditorPrivate
{
public:

  IGIVideoOverlayEditorPrivate();
  ~IGIVideoOverlayEditorPrivate();

  niftk::IGIVideoOverlayWidget* m_IGIVideoOverlayWidget;
  std::string m_FirstBackgroundColor;
  std::string m_SecondBackgroundColor;
  QScopedPointer<berry::IPartListener> m_PartListener;
};


/**
 * \class IGIOverlayWidgetPartListener
 * \brief Used to handle interaction with the contained overlay
 * editor widget when this IGIVideoOverlayEditor is opened/closed etc.
 */
struct IGIOverlayWidgetPartListener : public berry::IPartListener
{
  berryObjectMacro(IGIOverlayWidgetPartListener)

  //---------------------------------------------------------------------------
  IGIOverlayWidgetPartListener(IGIVideoOverlayEditorPrivate* dd)
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
    if (partRef->GetId() == IGIVideoOverlayEditor::EDITOR_ID)
    {
      IGIVideoOverlayEditor::Pointer editor = partRef->GetPart(false).Cast<IGIVideoOverlayEditor>();
      if (d->m_IGIVideoOverlayWidget == editor->GetIGIVideoOverlayWidget())
      {
        // Call editor to turn things off as the widget is being closed.
      }
    }
  }

  //---------------------------------------------------------------------------
  void PartHidden (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == IGIVideoOverlayEditor::EDITOR_ID)
    {
      IGIVideoOverlayEditor::Pointer editor = partRef->GetPart(false).Cast<IGIVideoOverlayEditor>();
      if (d->m_IGIVideoOverlayWidget == editor->GetIGIVideoOverlayWidget())
      {
        // Call editor to turn things off as the widget is being hidden.
      }
    }
  }

  //---------------------------------------------------------------------------
  void PartVisible (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == IGIVideoOverlayEditor::EDITOR_ID)
    {
      IGIVideoOverlayEditor::Pointer editor = partRef->GetPart(false).Cast<IGIVideoOverlayEditor>();
      if (d->m_IGIVideoOverlayWidget == editor->GetIGIVideoOverlayWidget())
      {
        // Call editor to turn things on as the widget is being made visible.
      }
    }
  }

private:

  IGIVideoOverlayEditorPrivate* const d;

};


//-----------------------------------------------------------------------------
IGIVideoOverlayEditorPrivate::IGIVideoOverlayEditorPrivate()
  : m_IGIVideoOverlayWidget(0)
  , m_PartListener(new IGIOverlayWidgetPartListener(this))
{}


//-----------------------------------------------------------------------------
IGIVideoOverlayEditorPrivate::~IGIVideoOverlayEditorPrivate()
{
}

//-----------------------------------------------------------------------------
IGIVideoOverlayEditor::IGIVideoOverlayEditor()
  : d(new IGIVideoOverlayEditorPrivate)
{
}


//-----------------------------------------------------------------------------
IGIVideoOverlayEditor::~IGIVideoOverlayEditor()
{
  this->GetSite()->GetPage()->RemovePartListener(d->m_PartListener.data());
}


//-----------------------------------------------------------------------------
niftk::IGIVideoOverlayWidget* IGIVideoOverlayEditor::GetIGIVideoOverlayWidget()
{
  return d->m_IGIVideoOverlayWidget;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *IGIVideoOverlayEditor::GetActiveQmitkRenderWindow() const
{
  return d->m_IGIVideoOverlayWidget->GetActiveQmitkRenderWindow();
}


//-----------------------------------------------------------------------------
QHash<QString, QmitkRenderWindow *> IGIVideoOverlayEditor::GetQmitkRenderWindows() const
{
  return d->m_IGIVideoOverlayWidget->GetQmitkRenderWindows();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *IGIVideoOverlayEditor::GetQmitkRenderWindow(const QString &id) const
{
  return d->m_IGIVideoOverlayWidget->GetQmitkRenderWindow(id);
}


//-----------------------------------------------------------------------------
mitk::Point3D IGIVideoOverlayEditor::GetSelectedPosition(const QString & id) const
{
  // Not implemented.
  mitk::Point3D point;
  point[0] = 0;
  point[1] = 0;
  point[2] = 0;
  return point;
}


//-----------------------------------------------------------------------------
void IGIVideoOverlayEditor::SetSelectedPosition(const mitk::Point3D &pos, const QString &id)
{
  // Not implemented.
}


//-----------------------------------------------------------------------------
void IGIVideoOverlayEditor::EnableDecorations(bool /*enable*/, const QStringList & /*decorations*/)
{
}


//-----------------------------------------------------------------------------
bool IGIVideoOverlayEditor::IsDecorationEnabled(const QString & /*decoration*/) const
{
  return false;
}


//-----------------------------------------------------------------------------
QStringList IGIVideoOverlayEditor::GetDecorations() const
{
  QStringList decorations;
  return decorations;
}


//-----------------------------------------------------------------------------
mitk::SlicesRotator* IGIVideoOverlayEditor::GetSlicesRotator() const
{
  return NULL;
}


//-----------------------------------------------------------------------------
mitk::SlicesSwiveller* IGIVideoOverlayEditor::GetSlicesSwiveller() const
{
  return NULL;
}


//-----------------------------------------------------------------------------
void IGIVideoOverlayEditor::EnableSlicingPlanes(bool /*enable*/)
{
}


//-----------------------------------------------------------------------------
bool IGIVideoOverlayEditor::IsSlicingPlanesEnabled() const
{
  return false;
}


//-----------------------------------------------------------------------------
void IGIVideoOverlayEditor::EnableLinkedNavigation(bool /*enable*/)
{
}


//-----------------------------------------------------------------------------
bool IGIVideoOverlayEditor::IsLinkedNavigationEnabled() const
{
  return false;
}


//-----------------------------------------------------------------------------
void IGIVideoOverlayEditor::CreateQtPartControl(QWidget* parent)
{
  if (d->m_IGIVideoOverlayWidget == 0)
  {

    mitk::DataStorage::Pointer ds = this->GetDataStorage();

    d->m_IGIVideoOverlayWidget = new niftk::IGIVideoOverlayWidget(parent);
    d->m_IGIVideoOverlayWidget->SetDataStorage(ds);

    QHBoxLayout* layout = new QHBoxLayout(parent);
    layout->setContentsMargins(0,0,0,0);
    layout->addWidget(d->m_IGIVideoOverlayWidget);

    this->GetSite()->GetPage()->AddPartListener(d->m_PartListener.data());

    QMetaObject::invokeMethod(this, "OnPreferencesChanged", Qt::QueuedConnection);

    this->RequestUpdate();

    // Finally: Listen to update pulse coming off of event bus. This pulse comes from the data manager updating.
    ctkServiceReference ref = niftk::IGIVideoOverlayEditorActivator::getContext()->getServiceReference<ctkEventAdmin>();
    if (ref)
    {
      ctkEventAdmin* eventAdmin = niftk::IGIVideoOverlayEditorActivator::getContext()->getService<ctkEventAdmin>(ref);
      
      ctkDictionary propertiesIGI;
      propertiesIGI[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
      eventAdmin->subscribeSlot(this, SLOT(OnIGIUpdate(ctkEvent)), propertiesIGI);
    }
  }
}


//-----------------------------------------------------------------------------
void IGIVideoOverlayEditor::OnPreferencesChanged()
{
  this->OnPreferencesChanged(dynamic_cast<berry::IBerryPreferences*>(this->GetPreferences().GetPointer()));
}


//-----------------------------------------------------------------------------
void IGIVideoOverlayEditor::OnPreferencesChanged(const berry::IBerryPreferences* prefs)
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
        d->m_IGIVideoOverlayWidget->DisableDepartmentLogo();
      }
      else
      {
        d->m_IGIVideoOverlayWidget->SetDepartmentLogoPath(departmentLogoLocation);
        d->m_IGIVideoOverlayWidget->EnableDepartmentLogo();
      }
      break;
    }
  }
 
  // Preferences for gradient background
  float color = 255.0;
  QString firstColorName = prefs->Get(IGIVideoOverlayEditorPreferencePage::FIRST_BACKGROUND_COLOUR, "");
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

  QString secondColorName = prefs->Get(IGIVideoOverlayEditorPreferencePage::SECOND_BACKGROUND_COLOUR, "");
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
  d->m_IGIVideoOverlayWidget->SetGradientBackgroundColors(upper, lower);
  d->m_IGIVideoOverlayWidget->EnableGradientBackground();
}


//-----------------------------------------------------------------------------
void IGIVideoOverlayEditor::SetFocus()
{
  if (d->m_IGIVideoOverlayWidget != 0)
  {
    d->m_IGIVideoOverlayWidget->setFocus();
  }
}


//-----------------------------------------------------------------------------
void IGIVideoOverlayEditor::OnIGIUpdate(const ctkEvent& event)
{
  d->m_IGIVideoOverlayWidget->Update();
}

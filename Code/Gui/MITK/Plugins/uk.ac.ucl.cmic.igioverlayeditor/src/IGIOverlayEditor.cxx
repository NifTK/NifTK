/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "IGIOverlayEditor.h"

#include <berryUIException.h>
#include <berryIWorkbenchPage.h>
#include <berryIPreferencesService.h>
#include <berryIPartListener.h>

#include <QWidget>

#include <mitkColorProperty.h>
#include <mitkGlobalInteraction.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateProperty.h>

#include <mitkDataStorageEditorInput.h>
#include <mitkIDataStorageService.h>

#include <QmitkSingleWidget.h>

const std::string IGIOverlayEditor::EDITOR_ID = "org.mitk.editors.igioverlayeditor";

class IGIOverlayEditorPrivate
{
public:

  IGIOverlayEditorPrivate();
  ~IGIOverlayEditorPrivate();

  QmitkSingleWidget* m_SingleWidget;
  std::string m_FirstBackgroundColor;
  std::string m_SecondBackgroundColor;
  berry::IPartListener::Pointer m_PartListener;
  QHash<QString, QmitkRenderWindow*> m_RenderWindows;
};

/**
 * \class QmitkSingleWidgetPartListener
 * \brief Used to handle interaction with the contained overlay editor widget when this IGIOverlayEditor is opened/closed etc.
 */
struct IGIOverlayWidgetPartListener : public berry::IPartListener
{
  berryObjectMacro(IGIOverlayWidgetPartListener)

  //---------------------------------------------------------------------------
  IGIOverlayWidgetPartListener(IGIOverlayEditorPrivate* dd)
    : d(dd)
  {}

  //---------------------------------------------------------------------------
  Events::Types GetPartEventTypes() const
  {
    return Events::CLOSED | Events::HIDDEN | Events::VISIBLE;
  }

  //---------------------------------------------------------------------------
  void PartClosed (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == IGIOverlayEditor::EDITOR_ID)
    {
      IGIOverlayEditor::Pointer editor = partRef->GetPart(false).Cast<IGIOverlayEditor>();
      if (d->m_SingleWidget == editor->GetSingleWidget())
      {
        // Call editor to turn things off as the widget is being closed.
      }
    }
  }

  //---------------------------------------------------------------------------
  void PartHidden (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == IGIOverlayEditor::EDITOR_ID)
    {
      IGIOverlayEditor::Pointer editor = partRef->GetPart(false).Cast<IGIOverlayEditor>();
      if (d->m_SingleWidget == editor->GetSingleWidget())
      {
        // Call editor to turn things off as the widget is being hidden.
      }
    }
  }

  //---------------------------------------------------------------------------
  void PartVisible (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == IGIOverlayEditor::EDITOR_ID)
    {
      IGIOverlayEditor::Pointer editor = partRef->GetPart(false).Cast<IGIOverlayEditor>();
      if (d->m_SingleWidget == editor->GetSingleWidget())
      {
        // Call editor to turn things on as the widget is being made visible.
      }
    }
  }

private:

  IGIOverlayEditorPrivate* const d;

};


//-----------------------------------------------------------------------------
IGIOverlayEditorPrivate::IGIOverlayEditorPrivate()
  : m_SingleWidget(0)
  , m_PartListener(new IGIOverlayWidgetPartListener(this))
{}


//-----------------------------------------------------------------------------
IGIOverlayEditorPrivate::~IGIOverlayEditorPrivate()
{
}

//-----------------------------------------------------------------------------
IGIOverlayEditor::IGIOverlayEditor()
  : d(new IGIOverlayEditorPrivate)
{
}


//-----------------------------------------------------------------------------
IGIOverlayEditor::~IGIOverlayEditor()
{
  this->GetSite()->GetPage()->RemovePartListener(d->m_PartListener);
}


//-----------------------------------------------------------------------------
QmitkSingleWidget* IGIOverlayEditor::GetSingleWidget()
{
  return d->m_SingleWidget;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *IGIOverlayEditor::GetActiveQmitkRenderWindow() const
{
  if (d->m_SingleWidget)
  {
    return d->m_SingleWidget->GetRenderWindow1();
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------------
QHash<QString, QmitkRenderWindow *> IGIOverlayEditor::GetQmitkRenderWindows() const
{
  return d->m_RenderWindows;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *IGIOverlayEditor::GetQmitkRenderWindow(const QString &id) const
{
  static bool alreadyWarned = false;

  if(!alreadyWarned)
  {
    MITK_WARN(id == "transversal") << "QmitkSingleWidgetEditor::GetRenderWindow(\"transversal\") is deprecated. Use \"axial\" instead.";
    alreadyWarned = true;
  }

  if (d->m_RenderWindows.contains(id))
  {
    return d->m_RenderWindows[id];
  }

  return 0;
}


//-----------------------------------------------------------------------------
mitk::Point3D IGIOverlayEditor::GetSelectedPosition(const QString & /*id*/) const
{
  return d->m_SingleWidget->GetCrossPosition();
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor::SetSelectedPosition(const mitk::Point3D &pos, const QString &/*id*/)
{
  d->m_SingleWidget->MoveCrossToPosition(pos);
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor::EnableDecorations(bool /*enable*/, const QStringList & /*decorations*/)
{
}


//-----------------------------------------------------------------------------
bool IGIOverlayEditor::IsDecorationEnabled(const QString & /*decoration*/) const
{
  return false;
}


//-----------------------------------------------------------------------------
QStringList IGIOverlayEditor::GetDecorations() const
{
  QStringList decorations;
  return decorations;
}


//-----------------------------------------------------------------------------
mitk::SlicesRotator* IGIOverlayEditor::GetSlicesRotator() const
{
  return NULL;
}


//-----------------------------------------------------------------------------
mitk::SlicesSwiveller* IGIOverlayEditor::GetSlicesSwiveller() const
{
  return NULL;
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor::EnableSlicingPlanes(bool /*enable*/)
{
}


//-----------------------------------------------------------------------------
bool IGIOverlayEditor::IsSlicingPlanesEnabled() const
{
  return false;
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor::EnableLinkedNavigation(bool /*enable*/)
{
}


//-----------------------------------------------------------------------------
bool IGIOverlayEditor::IsLinkedNavigationEnabled() const
{
  return false;
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor::CreateQtPartControl(QWidget* parent)
{
  if (d->m_SingleWidget == 0)
  {
    QHBoxLayout* layout = new QHBoxLayout(parent);
    layout->setContentsMargins(0,0,0,0);

    d->m_SingleWidget = new QmitkSingleWidget(parent);
    d->m_RenderWindows.insert("3d", d->m_SingleWidget->GetRenderWindow1());
    
    layout->addWidget(d->m_SingleWidget);

    mitk::DataStorage::Pointer ds = this->GetDataStorage();
    d->m_SingleWidget->SetDataStorage(ds);

    d->m_SingleWidget->GetRenderWindow1()->GetRenderer()->SetMapperID(mitk::BaseRenderer::Standard3D );

    this->GetSite()->GetPage()->AddPartListener(d->m_PartListener);

    berry::IPreferences::Pointer prefs = this->GetPreferences();
    this->OnPreferencesChanged(dynamic_cast<berry::IBerryPreferences*>(prefs.GetPointer()));

    this->RequestUpdate();
  }
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor::OnPreferencesChanged(const berry::IBerryPreferences* prefs)
{
  // Enable change of logo. If no DepartmentLogo was set explicitly, MBILogo is used.
  // Set new department logo by prefs->Set("DepartmentLogo", "PathToImage");

  std::vector<std::string> keys = prefs->Keys();
  
  for( unsigned int i = 0; i < keys.size(); ++i )
  {
    if( keys[i] == "DepartmentLogo")
    {
      std::string departmentLogoLocation = prefs->Get("DepartmentLogo", "");

      if (departmentLogoLocation.empty())
      {
        d->m_SingleWidget->DisableDepartmentLogo();
      }
      else
      {
        d->m_SingleWidget->SetDepartmentLogoPath(departmentLogoLocation.c_str());
        d->m_SingleWidget->EnableDepartmentLogo();
      }
      break;
    }
  }
 
  // Preferences for gradient background
  float color = 255.0;
  QString firstColorName = QString::fromStdString (prefs->GetByteArray("first background color", ""));
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

  QString secondColorName = QString::fromStdString (prefs->GetByteArray("second background color", ""));
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
  d->m_SingleWidget->SetGradientBackgroundColors(upper, lower);
  d->m_SingleWidget->EnableGradientBackground();
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor::SetFocus()
{
  if (d->m_SingleWidget != 0)
  {
    d->m_SingleWidget->setFocus();
  }
}

/*===================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center, 
Division of Medical and Biological Informatics.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without 
even the implied warranty of MERCHANTABILITY or FITNESS FOR 
A PARTICULAR PURPOSE.

See LICENSE.txt or http://www.mitk.org for details.

===================================================================*/

#include "QmitkSingleWidgetEditor.h"

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

#include <QmitkMouseModeSwitcher.h>
#include <QmitkSingleWidget.h>

#include <mbilogo.h>

class QmitkSingleWidgetEditorPrivate
{
public:

  QmitkSingleWidgetEditorPrivate();
  ~QmitkSingleWidgetEditorPrivate();

  QmitkSingleWidget* m_SingleWidget;
  QmitkMouseModeSwitcher* m_MouseModeToolbar;
  std::string m_FirstBackgroundColor;
  std::string m_SecondBackgroundColor;
  bool m_MenuWidgetsEnabled;
  berry::IPartListener::Pointer m_PartListener;

  QHash<QString, QmitkRenderWindow*> m_RenderWindows;
};

struct QmitkSingleWidgetPartListener : public berry::IPartListener
{
  berryObjectMacro(QmitkSingleWidgetPartListener)

  QmitkSingleWidgetPartListener(QmitkSingleWidgetEditorPrivate* dd)
    : d(dd)
  {}

  Events::Types GetPartEventTypes() const
  {
    return Events::CLOSED | Events::HIDDEN | Events::VISIBLE;
  }

  void PartClosed (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == QmitkSingleWidgetEditor::EDITOR_ID)
    {
      QmitkSingleWidgetEditor::Pointer singleWidgetEditor = partRef->GetPart(false).Cast<QmitkSingleWidgetEditor>();

      if (d->m_SingleWidget == singleWidgetEditor->GetSingleWidget())
      {
        d->m_SingleWidget->RemovePlanesFromDataStorage();
        singleWidgetEditor->RequestActivateMenuWidget(false);
      }
    }
  }

  void PartHidden (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == QmitkSingleWidgetEditor::EDITOR_ID)
    {
      QmitkSingleWidgetEditor::Pointer singleWidgetEditor = partRef->GetPart(false).Cast<QmitkSingleWidgetEditor>();

      if (d->m_SingleWidget == singleWidgetEditor->GetSingleWidget())
      {
        d->m_SingleWidget->RemovePlanesFromDataStorage();
        singleWidgetEditor->RequestActivateMenuWidget(false);
      }
    }
  }

  void PartVisible (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == QmitkSingleWidgetEditor::EDITOR_ID)
    {
      QmitkSingleWidgetEditor::Pointer singleWidgetEditor = partRef->GetPart(false).Cast<QmitkSingleWidgetEditor>();

      if (d->m_SingleWidget == singleWidgetEditor->GetSingleWidget())
      {
        d->m_SingleWidget->AddPlanesToDataStorage();
        singleWidgetEditor->RequestActivateMenuWidget(true);
      }
    }
  }

private:

  QmitkSingleWidgetEditorPrivate* const d;

};

QmitkSingleWidgetEditorPrivate::QmitkSingleWidgetEditorPrivate()
  : m_SingleWidget(0), m_MouseModeToolbar(0)
  , m_MenuWidgetsEnabled(false)
  , m_PartListener(new QmitkSingleWidgetPartListener(this))
{}

QmitkSingleWidgetEditorPrivate::~QmitkSingleWidgetEditorPrivate()
{
}

const std::string QmitkSingleWidgetEditor::EDITOR_ID = "uk.ac.ucl.cmic.editors.singlewidget";

QmitkSingleWidgetEditor::QmitkSingleWidgetEditor()
  : d(new QmitkSingleWidgetEditorPrivate)
{
}

QmitkSingleWidgetEditor::~QmitkSingleWidgetEditor()
{
  this->GetSite()->GetPage()->RemovePartListener(d->m_PartListener);
}

QmitkSingleWidget* QmitkSingleWidgetEditor::GetSingleWidget()
{
  return d->m_SingleWidget;
}

QmitkRenderWindow *QmitkSingleWidgetEditor::GetActiveQmitkRenderWindow() const
{
  if (d->m_SingleWidget) return d->m_SingleWidget->GetRenderWindow1();
  return 0;
}

QHash<QString, QmitkRenderWindow *> QmitkSingleWidgetEditor::GetQmitkRenderWindows() const
{
  return d->m_RenderWindows;
}

QmitkRenderWindow *QmitkSingleWidgetEditor::GetQmitkRenderWindow(const QString &id) const
{
  static bool alreadyWarned = false;

  if(!alreadyWarned)
  {
    MITK_WARN(id == "transversal") << "QmitkSingleWidgetEditor::GetRenderWindow(\"transversal\") is deprecated. Use \"axial\" instead.";
    alreadyWarned = true;
  }

  if (d->m_RenderWindows.contains(id))
    return d->m_RenderWindows[id];

  return 0;
}

mitk::Point3D QmitkSingleWidgetEditor::GetSelectedPosition(const QString & /*id*/) const
{
  return d->m_SingleWidget->GetCrossPosition();
}

void QmitkSingleWidgetEditor::SetSelectedPosition(const mitk::Point3D &pos, const QString &/*id*/)
{
  d->m_SingleWidget->MoveCrossToPosition(pos);
}

void QmitkSingleWidgetEditor::EnableDecorations(bool enable, const QStringList &decorations)
{
  if (decorations.isEmpty() || decorations.contains(DECORATION_BORDER))
  {
    enable ? d->m_SingleWidget->EnableColoredRectangles()
           : d->m_SingleWidget->DisableColoredRectangles();
  }
  if (decorations.isEmpty() || decorations.contains(DECORATION_LOGO))
  {
    enable ? d->m_SingleWidget->EnableDepartmentLogo()
           : d->m_SingleWidget->DisableDepartmentLogo();
  }
  if (decorations.isEmpty() || decorations.contains(DECORATION_MENU))
  {
    d->m_SingleWidget->ActivateMenuWidget(enable);
  }
  if (decorations.isEmpty() || decorations.contains(DECORATION_BACKGROUND))
  {
    enable ? d->m_SingleWidget->EnableGradientBackground()
           : d->m_SingleWidget->DisableGradientBackground();
  }
}

bool QmitkSingleWidgetEditor::IsDecorationEnabled(const QString &decoration) const
{
  if (decoration == DECORATION_BORDER)
  {
    return d->m_SingleWidget->IsColoredRectanglesEnabled();
  }
  else if (decoration == DECORATION_LOGO)
  {
    return d->m_SingleWidget->IsColoredRectanglesEnabled();
  }
  else if (decoration == DECORATION_MENU)
  {
    return d->m_SingleWidget->IsMenuWidgetEnabled();
  }
  else if (decoration == DECORATION_BACKGROUND)
  {
    return d->m_SingleWidget->GetGradientBackgroundFlag();
  }
  return false;
}

QStringList QmitkSingleWidgetEditor::GetDecorations() const
{
  QStringList decorations;
  decorations << DECORATION_BORDER << DECORATION_LOGO << DECORATION_MENU << DECORATION_BACKGROUND;
  return decorations;
}

mitk::SlicesRotator* QmitkSingleWidgetEditor::GetSlicesRotator() const
{
  return d->m_SingleWidget->GetSlicesRotator();
}

mitk::SlicesSwiveller* QmitkSingleWidgetEditor::GetSlicesSwiveller() const
{
  return d->m_SingleWidget->GetSlicesSwiveller();
}

void QmitkSingleWidgetEditor::EnableSlicingPlanes(bool enable)
{
  d->m_SingleWidget->SetWidgetPlanesVisibility(enable);
}

bool QmitkSingleWidgetEditor::IsSlicingPlanesEnabled() const
{
  mitk::DataNode::Pointer node = this->d->m_SingleWidget->GetWidgetPlane1();
  if (node.IsNotNull())
  {
    bool visible = false;
    node->GetVisibility(visible, 0);
    return visible;
  }
  else
  {
    return false;
  }
}

void QmitkSingleWidgetEditor::EnableLinkedNavigation(bool enable)
{
  enable ? d->m_SingleWidget->EnableNavigationControllerEventListening()
         : d->m_SingleWidget->DisableNavigationControllerEventListening();
}

bool QmitkSingleWidgetEditor::IsLinkedNavigationEnabled() const
{
  return d->m_SingleWidget->IsCrosshairNavigationEnabled();
}

void QmitkSingleWidgetEditor::CreateQtPartControl(QWidget* parent)
{
  if (d->m_SingleWidget == 0)
  {
    QHBoxLayout* layout = new QHBoxLayout(parent);
    layout->setContentsMargins(0,0,0,0);

    if (d->m_MouseModeToolbar == NULL)
    {
      d->m_MouseModeToolbar = new QmitkMouseModeSwitcher(parent); // delete by Qt via parent
      layout->addWidget(d->m_MouseModeToolbar);
    }

    d->m_SingleWidget = new QmitkSingleWidget(parent);

    d->m_RenderWindows.insert("3d", d->m_SingleWidget->GetRenderWindow1());
    
    d->m_MouseModeToolbar->setMouseModeSwitcher( d->m_SingleWidget->GetMouseModeSwitcher() );
    connect( d->m_MouseModeToolbar, SIGNAL( MouseModeSelected(mitk::MouseModeSwitcher::MouseMode) ),
      d->m_SingleWidget, SLOT( MouseModeSelected(mitk::MouseModeSwitcher::MouseMode) ) );

    layout->addWidget(d->m_SingleWidget);

    mitk::DataStorage::Pointer ds = this->GetDataStorage();

    // Tell the multiWidget which (part of) the tree to render
    d->m_SingleWidget->SetDataStorage(ds);

    // Initialize views as axial, sagittal, coronar to all data objects in DataStorage
    // (from top-left to bottom)
    mitk::TimeSlicedGeometry::Pointer geo = ds->ComputeBoundingGeometry3D(ds->GetAll());
    mitk::RenderingManager::GetInstance()->InitializeViews(geo);

    // Initialize bottom-right view as 3D view
    d->m_SingleWidget->GetRenderWindow1()->GetRenderer()->SetMapperID(
      mitk::BaseRenderer::Standard3D );

    // Enable standard handler for levelwindow-slider
    d->m_SingleWidget->EnableStandardLevelWindow();

    // Add the displayed views to the tree to see their positions
    // in 2D and 3D
    d->m_SingleWidget->AddDisplayPlaneSubTree();

    d->m_SingleWidget->EnableNavigationControllerEventListening();

    // Store the initial visibility status of the menu widget.
    d->m_MenuWidgetsEnabled = d->m_SingleWidget->IsMenuWidgetEnabled();

    this->GetSite()->GetPage()->AddPartListener(d->m_PartListener);

    berry::IPreferences::Pointer prefs = this->GetPreferences();
    this->OnPreferencesChanged(dynamic_cast<berry::IBerryPreferences*>(prefs.GetPointer()));

    this->RequestUpdate();
  }
}

void QmitkSingleWidgetEditor::OnPreferencesChanged(const berry::IBerryPreferences* prefs)
{
  // Enable change of logo. If no DepartmentLogo was set explicitly, MBILogo is used.
  // Set new department logo by prefs->Set("DepartmentLogo", "PathToImage");

  std::vector<std::string> keys = prefs->Keys();
  
  for( int i = 0; i < keys.size(); ++i )
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
 
  // preferences for gradient background
  float color = 255.0;
  QString firstColorName = QString::fromStdString (prefs->GetByteArray("first background color", ""));
  QColor firstColor(firstColorName);
  mitk::Color upper;
  if (firstColorName=="") // default values
  {
    upper[0] = 0.1;
    upper[1] = 0.1;
    upper[2] = 0.1;
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
    lower[0] = 0.5;
    lower[1] = 0.5;
    lower[2] = 0.5;
  }
  else
  {
    lower[0] = secondColor.red() / color;
    lower[1] = secondColor.green() / color;
    lower[2] = secondColor.blue() / color;
  }
  d->m_SingleWidget->SetGradientBackgroundColors(upper, lower);
  d->m_SingleWidget->EnableGradientBackground();

  // Set preferences respecting zooming and padding
  bool constrainedZooming = prefs->GetBool("Use constrained zooming and padding", false);

  mitk::RenderingManager::GetInstance()->SetConstrainedPaddingZooming(constrainedZooming);

  mitk::NodePredicateNot::Pointer pred
    = mitk::NodePredicateNot::New(mitk::NodePredicateProperty::New("includeInBoundingBox"
    , mitk::BoolProperty::New(false)));

  mitk::DataStorage::SetOfObjects::ConstPointer rs = this->GetDataStorage()->GetSubset(pred);
  // calculate bounding geometry of these nodes

  mitk::TimeSlicedGeometry::Pointer bounds = this->GetDataStorage()->ComputeBoundingGeometry3D(rs, "visible");


  // initialize the views to the bounding geometry
  mitk::RenderingManager::GetInstance()->InitializeViews(bounds);

  mitk::RenderingManager::GetInstance()->RequestUpdateAll();

  // level window setting
  bool showLevelWindowWidget = prefs->GetBool("Show level/window widget", true);
  if (showLevelWindowWidget)
  {
    d->m_SingleWidget->EnableStandardLevelWindow();
  }
  else
  {
    d->m_SingleWidget->DisableStandardLevelWindow();
  }

  // mouse modes toolbar
  bool newMode = prefs->GetBool("PACS like mouse interaction", false);
  d->m_MouseModeToolbar->setVisible( newMode );
  d->m_SingleWidget->GetMouseModeSwitcher()->SetInteractionScheme( newMode ? mitk::MouseModeSwitcher::PACS : mitk::MouseModeSwitcher::MITK );
}

void QmitkSingleWidgetEditor::SetFocus()
{
  if (d->m_SingleWidget != 0)
    d->m_SingleWidget->setFocus();
}

void QmitkSingleWidgetEditor::RequestActivateMenuWidget(bool on)
{
  if (d->m_SingleWidget)
  {
    if (on)
    {
      d->m_SingleWidget->ActivateMenuWidget(d->m_MenuWidgetsEnabled);
    }
    else
    {
      d->m_MenuWidgetsEnabled = d->m_SingleWidget->IsMenuWidgetEnabled();
      d->m_SingleWidget->ActivateMenuWidget(false);
    }
  }
}

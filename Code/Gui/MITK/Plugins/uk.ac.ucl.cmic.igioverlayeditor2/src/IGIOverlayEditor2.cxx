/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "IGIOverlayEditor2.h"

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

#include <OverlayEditor2/QmitkIGIOverlayEditor2.h>
#include <internal/IGIOverlayEditor2PreferencePage.h>
#include <internal/IGIOverlayEditor2Activator.h>

const char* IGIOverlayEditor2::EDITOR_ID = "org.mitk.editors.igioverlayeditor2";


/**
 * \class IGIOverlayEditor2Private
 * \brief PIMPL pattern implementation of IGIOverlayEditor2.
 */
class IGIOverlayEditor2Private
{
public:

  IGIOverlayEditor2Private();
  ~IGIOverlayEditor2Private();

  QmitkIGIOverlayEditor2* m_IGIOverlayEditor2;
  std::string m_FirstBackgroundColor;
  std::string m_SecondBackgroundColor;
  berry::IPartListener::Pointer m_PartListener;
};


/**
 * \class IGIOverlayWidgetPartListener
 * \brief Used to handle interaction with the contained overlay
 * editor widget when this IGIOverlayEditor is opened/closed etc.
 */
struct IGIOverlay2WidgetPartListener : public berry::IPartListener
{
  berryObjectMacro(IGIOverlay2WidgetPartListener)

  //---------------------------------------------------------------------------
  IGIOverlay2WidgetPartListener(IGIOverlayEditor2Private* dd)
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
    if (partRef->GetId() == IGIOverlayEditor2::EDITOR_ID)
    {
      IGIOverlayEditor2::Pointer editor = partRef->GetPart(false).Cast<IGIOverlayEditor2>();
      if (d->m_IGIOverlayEditor2 == editor->GetIGIOverlayEditor2())
      {
        // Call editor to turn things off as the widget is being closed.
      }
    }
  }

  //---------------------------------------------------------------------------
  void PartHidden (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == IGIOverlayEditor2::EDITOR_ID)
    {
      IGIOverlayEditor2::Pointer editor = partRef->GetPart(false).Cast<IGIOverlayEditor2>();
      if (d->m_IGIOverlayEditor2 == editor->GetIGIOverlayEditor2())
      {
        // Call editor to turn things off as the widget is being hidden.
      }
    }
  }

  //---------------------------------------------------------------------------
  void PartVisible (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == IGIOverlayEditor2::EDITOR_ID)
    {
      IGIOverlayEditor2::Pointer editor = partRef->GetPart(false).Cast<IGIOverlayEditor2>();
      if (d->m_IGIOverlayEditor2 == editor->GetIGIOverlayEditor2())
      {
        // Call editor to turn things on as the widget is being made visible.
      }
    }
  }

private:

  IGIOverlayEditor2Private* const d;

};


//-----------------------------------------------------------------------------
IGIOverlayEditor2Private::IGIOverlayEditor2Private()
  : m_IGIOverlayEditor2(0)
  , m_PartListener(new IGIOverlay2WidgetPartListener(this))
{}


//-----------------------------------------------------------------------------
IGIOverlayEditor2Private::~IGIOverlayEditor2Private()
{
}

//-----------------------------------------------------------------------------
IGIOverlayEditor2::IGIOverlayEditor2()
  : d(new IGIOverlayEditor2Private)
{
}


//-----------------------------------------------------------------------------
IGIOverlayEditor2::~IGIOverlayEditor2()
{
  this->GetSite()->GetPage()->RemovePartListener(d->m_PartListener);
}


//-----------------------------------------------------------------------------
QmitkIGIOverlayEditor2* IGIOverlayEditor2::GetIGIOverlayEditor2()
{
  return d->m_IGIOverlayEditor2;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *IGIOverlayEditor2::GetActiveQmitkRenderWindow() const
{
  return d->m_IGIOverlayEditor2->GetActiveQmitkRenderWindow();
}


//-----------------------------------------------------------------------------
QHash<QString, QmitkRenderWindow *> IGIOverlayEditor2::GetQmitkRenderWindows() const
{
  return d->m_IGIOverlayEditor2->GetQmitkRenderWindows();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *IGIOverlayEditor2::GetQmitkRenderWindow(const QString &id) const
{
  return d->m_IGIOverlayEditor2->GetQmitkRenderWindow(id);
}


//-----------------------------------------------------------------------------
mitk::Point3D IGIOverlayEditor2::GetSelectedPosition(const QString & id) const
{
  // Not implemented.
  mitk::Point3D point;
  point[0] = 0;
  point[1] = 0;
  point[2] = 0;
  return point;
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::SetSelectedPosition(const mitk::Point3D &pos, const QString &id)
{
  // Not implemented.
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::EnableDecorations(bool /*enable*/, const QStringList & /*decorations*/)
{
}


//-----------------------------------------------------------------------------
bool IGIOverlayEditor2::IsDecorationEnabled(const QString & /*decoration*/) const
{
  return false;
}


//-----------------------------------------------------------------------------
QStringList IGIOverlayEditor2::GetDecorations() const
{
  QStringList decorations;
  return decorations;
}


//-----------------------------------------------------------------------------
mitk::SlicesRotator* IGIOverlayEditor2::GetSlicesRotator() const
{
  return NULL;
}


//-----------------------------------------------------------------------------
mitk::SlicesSwiveller* IGIOverlayEditor2::GetSlicesSwiveller() const
{
  return NULL;
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::EnableSlicingPlanes(bool /*enable*/)
{
}


//-----------------------------------------------------------------------------
bool IGIOverlayEditor2::IsSlicingPlanesEnabled() const
{
  return false;
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::EnableLinkedNavigation(bool /*enable*/)
{
}


//-----------------------------------------------------------------------------
bool IGIOverlayEditor2::IsLinkedNavigationEnabled() const
{
  return false;
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::CreateQtPartControl(QWidget* parent)
{
  if (d->m_IGIOverlayEditor2 == 0)
  {
    QHBoxLayout* layout = new QHBoxLayout(parent);
    layout->setContentsMargins(0,0,0,0);

    d->m_IGIOverlayEditor2 = new QmitkIGIOverlayEditor2(parent);
    layout->addWidget(d->m_IGIOverlayEditor2);

    mitk::DataStorage::Pointer ds = this->GetDataStorage();
    d->m_IGIOverlayEditor2->SetDataStorage(ds);

    this->GetSite()->GetPage()->AddPartListener(d->m_PartListener);

    QMetaObject::invokeMethod(this, "OnPreferencesChanged", Qt::QueuedConnection);

    this->RequestUpdate();

    // Finally: Listen to update pulse coming off of event bus. This pulse comes from the data manager updating.
    ctkServiceReference ref = mitk::IGIOverlayEditor2Activator::getContext()->getServiceReference<ctkEventAdmin>();
    if (ref)
    {
      ctkEventAdmin* eventAdmin = mitk::IGIOverlayEditor2Activator::getContext()->getService<ctkEventAdmin>(ref);
      
      ctkDictionary propertiesIGI;
      propertiesIGI[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
      eventAdmin->subscribeSlot(this, SLOT(OnIGIUpdate(ctkEvent)), propertiesIGI);
      
      ctkDictionary propertiesTrackedImage;
      propertiesTrackedImage[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGITRACKEDIMAGEUPDATE";
      eventAdmin->subscribeSlot(this, SLOT(OnTrackedImageUpdate(ctkEvent)), propertiesTrackedImage, Qt::DirectConnection);

      ctkDictionary propertiesRecordingStarted;
      propertiesRecordingStarted[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIRECORDINGSTARTED";
      eventAdmin->subscribeSlot(this, SLOT(OnRecordingStarted(ctkEvent)), propertiesRecordingStarted);
    }
  }
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::OnPreferencesChanged()
{
  this->OnPreferencesChanged(dynamic_cast<berry::IBerryPreferences*>(this->GetPreferences().GetPointer()));
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::OnPreferencesChanged(const berry::IBerryPreferences* prefs)
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
        d->m_IGIOverlayEditor2->DisableDepartmentLogo();
      }
      else
      {
        d->m_IGIOverlayEditor2->SetDepartmentLogoPath(departmentLogoLocation);
        d->m_IGIOverlayEditor2->EnableDepartmentLogo();
      }
      break;
    }
  }
 
  // Preferences for gradient background
  float color = 255.0;
  QString firstColorName = QString::fromStdString (prefs->GetByteArray(IGIOverlayEditor2PreferencePage::FIRST_BACKGROUND_COLOUR, ""));
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

  QString secondColorName = QString::fromStdString (prefs->GetByteArray(IGIOverlayEditor2PreferencePage::SECOND_BACKGROUND_COLOUR, ""));
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
  d->m_IGIOverlayEditor2->SetGradientBackgroundColors(upper, lower);
  d->m_IGIOverlayEditor2->EnableGradientBackground();

  std::string calibrationFileName = prefs->Get(IGIOverlayEditor2PreferencePage::CALIBRATION_FILE_NAME, "");
  d->m_IGIOverlayEditor2->SetCalibrationFileName(calibrationFileName);
  d->m_IGIOverlayEditor2->SetCameraTrackingMode(prefs->GetBool(IGIOverlayEditor2PreferencePage::CAMERA_TRACKING_MODE, true));
  d->m_IGIOverlayEditor2->SetClipToImagePlane(prefs->GetBool(IGIOverlayEditor2PreferencePage::CLIP_TO_IMAGE_PLANE, true));
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::SetFocus()
{
  if (d->m_IGIOverlayEditor2 != 0)
  {
    d->m_IGIOverlayEditor2->setFocus();
  }
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::OnIGIUpdate(const ctkEvent& event)
{
  d->m_IGIOverlayEditor2->Update();
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::OnTrackedImageUpdate(const ctkEvent& event)
{
  d->m_IGIOverlayEditor2->Update();
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::WriteCurrentConfig(const QString& directory) const
{
  QFile   infoFile(directory + QDir::separator() + EDITOR_ID + ".txt");
  bool opened = infoFile.open(QIODevice::ReadWrite | QIODevice::Text | QIODevice::Append);
  if (opened)
  {
    QTextStream   info(&infoFile);
    info.setCodec("UTF-8");
    info << "START: " << QDateTime::currentDateTime().toString() << "\n";
    info << "calibfile=" << QString::fromStdString(d->m_IGIOverlayEditor2->GetCalibrationFileName()) << "\n";
  }
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2::OnRecordingStarted(const ctkEvent& event)
{
  QString   directory = event.getProperty("directory").toString();
  if (!directory.isEmpty())
  {
    try
    {
      WriteCurrentConfig(directory);
    }
    catch (...)
    {
      MITK_ERROR << "Caught exception while writing info file! Ignoring it and aborting info file.";
    }
  }
  else
  {
    MITK_WARN << "Received igi-recording-started event without directory information! Ignoring it.";
  }
}

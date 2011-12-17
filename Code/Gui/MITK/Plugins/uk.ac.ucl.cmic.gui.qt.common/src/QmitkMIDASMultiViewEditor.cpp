/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-15 15:03:56 +0000 (Thu, 15 Dec 2011) $
 Revision          : $Revision: 8030 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkMIDASMultiViewEditor.h"

#include <berryUIException.h>
#include <berryIWorkbenchPage.h>
#include <berryIPreferencesService.h>

#include <QWidget>
#include <QGridLayout>

#include "mitkGlobalInteraction.h"
#include "mitkIDataStorageService.h"
#include "mitkNodePredicateNot.h"
#include "mitkNodePredicateProperty.h"
#include "mitkMIDASDataStorageEditorInput.h"
#include "internal/CommonActivator.h"

#include "QmitkMIDASMultiViewWidget.h"
#include "QmitkMIDASMultiViewVisibilityManager.h"
#include "QmitkMIDASMultiViewEditorPreferencePage.h"

// CTK
#include "ctkPluginContext.h"
#include "service/event/ctkEvent.h"
#include "service/event/ctkEventConstants.h"

const std::string QmitkMIDASMultiViewEditor::EDITOR_ID = "org.mitk.editors.midasmultiview";

QmitkMIDASMultiViewEditor::QmitkMIDASMultiViewEditor()
 : m_MIDASMultiViewWidget(NULL)
, m_MidasMultiViewVisibilityManager(NULL)
, m_Context(NULL)
, m_EventAdmin(NULL)
{
}

QmitkMIDASMultiViewEditor::QmitkMIDASMultiViewEditor(const QmitkMIDASMultiViewEditor& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}

QmitkMIDASMultiViewEditor::~QmitkMIDASMultiViewEditor()
{
  // we need to wrap the RemovePartListener call inside a
  // register/unregister block to prevent infinite recursion
  // due to the destruction of temporary smartpointer to this
  this->Register();
  this->GetSite()->GetPage()->RemovePartListener(berry::IPartListener::Pointer(this));
  this->UnRegister(false);

  if (m_MidasMultiViewVisibilityManager != NULL)
  {
    delete m_MidasMultiViewVisibilityManager;
  }
}

QmitkMIDASMultiViewWidget* QmitkMIDASMultiViewEditor::GetMIDASMultiViewWidget()
{
  return m_MIDASMultiViewWidget;
}

void QmitkMIDASMultiViewEditor::Init(berry::IEditorSite::Pointer site, berry::IEditorInput::Pointer input)
{
  if (input.Cast<mitk::MIDASDataStorageEditorInput>().IsNull())
  {
    throw berry::PartInitException("Invalid Input: Must be MIDASDataStorageEditorInput");
  }
  this->SetSite(site);
  this->SetInput(input);
}

mitk::DataStorage::Pointer QmitkMIDASMultiViewEditor::GetDataStorage() const
{
  mitk::IDataStorageService::Pointer service =
    berry::Platform::GetServiceRegistry().GetServiceById<mitk::IDataStorageService>(mitk::IDataStorageService::ID);

  if (service.IsNotNull())
  {
    return service->GetDefaultDataStorage()->GetDataStorage();
  }

  return 0;
}

void QmitkMIDASMultiViewEditor::CreateQtPartControl(QWidget* parent)
{
  if (m_MIDASMultiViewWidget == NULL)
  {
    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();

    berry::IPreferencesService::Pointer prefService = berry::Platform::GetServiceRegistry().GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);
    berry::IBerryPreferences::Pointer prefs = (prefService->GetSystemPreferences()->Node(EDITOR_ID)).Cast<berry::IBerryPreferences>();
    assert( prefs );

    QmitkMIDASMultiViewVisibilityManager::MIDASDefaultInterpolationType defaultInterpolation =
        (QmitkMIDASMultiViewVisibilityManager::MIDASDefaultInterpolationType)(prefs->GetInt(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_IMAGE_INTERPOLATION, 0));
    QmitkMIDASMultiViewVisibilityManager::MIDASDefaultOrientationType defaultOrientation =
        (QmitkMIDASMultiViewVisibilityManager::MIDASDefaultOrientationType)(prefs->GetInt(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_ORIENTATION, 0));
    int defaultNumberOfRows = prefs->GetInt(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_NUMBER_ROWS, 1);
    int defaultNumberOfColumns = prefs->GetInt(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_NUMBER_COLUMNS, 1);

    m_MidasMultiViewVisibilityManager = new QmitkMIDASMultiViewVisibilityManager(dataStorage);
    m_MidasMultiViewVisibilityManager->SetDefaultInterpolationType(defaultInterpolation);
    m_MidasMultiViewVisibilityManager->SetDefaultOrientationType(defaultOrientation);

    // Create the QmitkMIDASMultiViewWidget
    m_MIDASMultiViewWidget = new QmitkMIDASMultiViewWidget(
        m_MidasMultiViewVisibilityManager,
        defaultNumberOfRows,
        defaultNumberOfColumns,
        parent);

    // Connect slots
    connect(m_MIDASMultiViewWidget, SIGNAL(UpdateMIDASViewingControlsRange(UpdateMIDASViewingControlsRangeInfo)), this, SLOT(OnUpdateMIDASViewingControlsRange(UpdateMIDASViewingControlsRangeInfo)));
    connect(m_MIDASMultiViewWidget, SIGNAL(UpdateMIDASViewingControlsValues(UpdateMIDASViewingControlsInfo)), this, SLOT(OnUpdateMIDASViewingControlsValues(UpdateMIDASViewingControlsInfo)));

    QGridLayout *gridLayout = new QGridLayout(parent);
    gridLayout->addWidget(m_MIDASMultiViewWidget, 0, 0);
    gridLayout->setContentsMargins(0, 0, 0, 0);
    gridLayout->setSpacing(0);

    prefs->OnChanged.AddListener( berry::MessageDelegate1<QmitkMIDASMultiViewEditor, const berry::IBerryPreferences*>( this, &QmitkMIDASMultiViewEditor::OnPreferencesChanged ) );
    this->OnPreferencesChanged(prefs.GetPointer());

    m_Context = mitk::CommonActivator::GetPluginContext();
    m_EventAdminRef = m_Context->getServiceReference<ctkEventAdmin>();
    m_EventAdmin = m_Context->getService<ctkEventAdmin>(m_EventAdminRef);

    ctkDictionary props;
    props[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/midasnavigationview/*";
    m_Context->registerService<ctkEventHandler>(this, props);
  }
}

berry::IPartListener::Events::Types QmitkMIDASMultiViewEditor::GetPartEventTypes() const
{
  return Events::CLOSED | Events::HIDDEN | Events::VISIBLE;
}

void QmitkMIDASMultiViewEditor::PartClosed( berry::IWorkbenchPartReference::Pointer partRef )
{
  if (partRef->GetId() == QmitkMIDASMultiViewEditor::EDITOR_ID)
  {
    QmitkMIDASMultiViewEditor::Pointer midasMultiViewEditor = partRef->GetPart(false).Cast<QmitkMIDASMultiViewEditor>();

    if (m_MIDASMultiViewWidget == midasMultiViewEditor->GetMIDASMultiViewWidget())
    {
      m_MIDASMultiViewWidget->setEnabled(false);
    }
  }
}

void QmitkMIDASMultiViewEditor::PartVisible( berry::IWorkbenchPartReference::Pointer partRef )
{
  if (partRef->GetId() == QmitkMIDASMultiViewEditor::EDITOR_ID)
  {
    QmitkMIDASMultiViewEditor::Pointer midasMultiViewEditor = partRef->GetPart(false).Cast<QmitkMIDASMultiViewEditor>();

    if (m_MIDASMultiViewWidget == midasMultiViewEditor->GetMIDASMultiViewWidget())
    {
      m_MIDASMultiViewWidget->setEnabled(true);
    }
  }
}

void QmitkMIDASMultiViewEditor::PartHidden( berry::IWorkbenchPartReference::Pointer partRef )
{
  if (partRef->GetId() == QmitkMIDASMultiViewEditor::EDITOR_ID)
  {
    QmitkMIDASMultiViewEditor::Pointer midasMultiViewEditor = partRef->GetPart(false).Cast<QmitkMIDASMultiViewEditor>();
    if (m_MIDASMultiViewWidget == midasMultiViewEditor->GetMIDASMultiViewWidget())
    {
      // Do stuff
    }
  }
}

void QmitkMIDASMultiViewEditor::SetFocus()
{
  if (m_MIDASMultiViewWidget != NULL)
  {
    m_MIDASMultiViewWidget->setFocus();
  }
}

void QmitkMIDASMultiViewEditor::OnPreferencesChanged( const berry::IBerryPreferences* prefs )
{
  if (m_MIDASMultiViewWidget != NULL)
  {
    QString backgroundColourName = QString::fromStdString (prefs->GetByteArray(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_BACKGROUND_COLOUR, ""));
    QColor backgroundColour(backgroundColourName);
    mitk::Color bgColour;
    if (backgroundColourName=="") // default values
    {
      bgColour[0] = 1;
      bgColour[1] = 0.980392;
      bgColour[2] = 0.941176;
    }
    else
    {
      bgColour[0] = backgroundColour.red() / 255.0;
      bgColour[1] = backgroundColour.green() / 255.0;
      bgColour[2] = backgroundColour.blue() / 255.0;
    }
    m_MIDASMultiViewWidget->SetBackgroundColour(bgColour);
    m_MIDASMultiViewWidget->SetDefaultInterpolationType((QmitkMIDASMultiViewVisibilityManager::MIDASDefaultInterpolationType)(prefs->GetInt(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_IMAGE_INTERPOLATION, 0)));
    m_MIDASMultiViewWidget->SetDefaultOrientationType((QmitkMIDASMultiViewVisibilityManager::MIDASDefaultOrientationType)(prefs->GetInt(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_ORIENTATION, 0)));
  }
}

void QmitkMIDASMultiViewEditor::handleEvent(const ctkEvent& event)
{
  try
  {
    if (m_MIDASMultiViewWidget != NULL)
    {
      QString topic = event.getProperty("topic").toString();
      QVariant value = event.getProperty("value");

      if (topic == "slice")
      {
        m_MIDASMultiViewWidget->SetSelectedWindowSliceNumber(value.toInt());
      }
      else if (topic == "magnification")
      {
        m_MIDASMultiViewWidget->SetSelectedWindowMagnification(value.toInt());
      }
      else if (topic == "orientation")
      {
        if (value == "axial")
        {
          m_MIDASMultiViewWidget->SetSelectedWindowToAxial();
        }
        else if (value == "sagittal")
        {
          m_MIDASMultiViewWidget->SetSelectedWindowToSagittal();
        }
        else if (value == "coronal")
        {
          m_MIDASMultiViewWidget->SetSelectedWindowToCoronal();
        }
      }
    }
  }
  catch (const ctkRuntimeException& e)
  {
    MITK_ERROR << "QmitkMIDASMultiViewEditor::handleEvent, failed with:" << e.what() \
        << ", caused by " << e.getCause().toLocal8Bit().constData() \
        << std::endl;
  }
}

void QmitkMIDASMultiViewEditor::OnUpdateMIDASViewingControlsRange(UpdateMIDASViewingControlsRangeInfo rangeInfo)
{
  try
  {
    ctkDictionary message;
    message["type"] = "UpdateMIDASViewingControlsRangeInfo";
    message["min_slice"] = rangeInfo.minSlice;
    message["max_slice"] = rangeInfo.maxSlice;
    message["min_magnification"] = rangeInfo.minMagnification;
    message["max_magnification"] = rangeInfo.maxMagnification;

    ctkEvent event("uk/ac/ucl/cmic/niftkQmitkExt/QmitkMIDASMultiViewEditor/OnUpdateMIDASViewingControls", message);
    m_EventAdmin->sendEvent(event);
  }
  catch (const ctkRuntimeException& e)
  {
    MITK_ERROR << "QmitkMIDASMultiViewEditor::OnUpdateMIDASViewingControls, failed with:" << e.what() \
        << ", caused by " << e.getCause().toLocal8Bit().constData() \
        << std::endl;
  }
}

void QmitkMIDASMultiViewEditor::OnUpdateMIDASViewingControlsValues(UpdateMIDASViewingControlsInfo info)
{
  try
  {
    ctkDictionary message;
    message["type"] = "UpdateMIDASViewingControlsInfo";
    message["current_slice"] = info.currentSlice;
    message["current_magnification"] = info.currentMagnification;
    if (info.isAxial)
    {
      message["orientation"] = "axial";
    }
    else if (info.isSagittal)
    {
      message["orientation"] = "sagittal";
    }
    else if (info.isCoronal)
    {
      message["orientation"] = "coronal";
    }

    ctkEvent event("uk/ac/ucl/cmic/niftkQmitkExt/QmitkMIDASMultiViewEditor/OnUpdateMIDASViewingControls", message);
    m_EventAdmin->sendEvent(event);
  }
  catch (const ctkRuntimeException& e)
  {
    MITK_ERROR << "QmitkMIDASMultiViewEditor::OnUpdateMIDASViewingControls, failed with:" << e.what() \
        << ", caused by " << e.getCause().toLocal8Bit().constData() \
        << std::endl;
  }
}

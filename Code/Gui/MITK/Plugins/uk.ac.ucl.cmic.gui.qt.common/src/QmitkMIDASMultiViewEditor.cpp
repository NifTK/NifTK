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
 :
  m_KeyPressStateMachine(NULL)
, m_MIDASMultiViewWidget(NULL)
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
        (QmitkMIDASMultiViewVisibilityManager::MIDASDefaultInterpolationType)(prefs->GetInt(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_IMAGE_INTERPOLATION, 2));
    QmitkMIDASMultiViewVisibilityManager::MIDASDefaultOrientationType defaultOrientation =
        (QmitkMIDASMultiViewVisibilityManager::MIDASDefaultOrientationType)(prefs->GetInt(QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_ORIENTATION, 2));
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

    this->GetSite()->GetPage()->AddPartListener(berry::IPartListener::Pointer(this));

    QGridLayout *gridLayout = new QGridLayout(parent);
    gridLayout->addWidget(m_MIDASMultiViewWidget, 0, 0);
    gridLayout->setContentsMargins(0, 0, 0, 0);
    gridLayout->setSpacing(0);

    prefs->OnChanged.AddListener( berry::MessageDelegate1<QmitkMIDASMultiViewEditor, const berry::IBerryPreferences*>( this, &QmitkMIDASMultiViewEditor::OnPreferencesChanged ) );
    this->OnPreferencesChanged(prefs.GetPointer());

    // Connect slots
    connect(m_MIDASMultiViewWidget, SIGNAL(UpdateMIDASViewingControlsValues(UpdateMIDASViewingControlsInfo)), this, SLOT(OnUpdateMIDASViewingControlsValues(UpdateMIDASViewingControlsInfo)));

    m_Context = mitk::CommonActivator::GetPluginContext();
    m_EventAdminRef = m_Context->getServiceReference<ctkEventAdmin>();
    m_EventAdmin = m_Context->getService<ctkEventAdmin>(m_EventAdminRef);

    m_EventAdmin->publishSignal(this, SIGNAL(UpdateMIDASViewingControlsValues(ctkDictionary)),
                              "uk/ac/ucl/cmic/gui/qt/common/QmitkMIDASMultiViewEditor/OnUpdateMIDASViewingControlsValues", Qt::QueuedConnection);

    m_EventAdmin->publishSignal(this, SIGNAL(PartStatusChanged(ctkDictionary)),
                              "uk/ac/ucl/cmic/gui/qt/common/QmitkMIDASMultiViewEditor/PartStatusChanged", Qt::QueuedConnection);

    // Dummy call, i think there is a threading / race condition, so Im trying to initialise this early.
    mitk::GlobalInteraction::GetInstance()->GetFocusManager();

    // Create/Connect the state machine
    m_KeyPressStateMachine = mitk::MIDASKeyPressStateMachine::New("MIDASKeyPressStateMachine", m_MIDASMultiViewWidget);
    mitk::GlobalInteraction::GetInstance()->AddListener( m_KeyPressStateMachine );

    ctkDictionary propsForSlot;
    propsForSlot[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/midasnavigationview/*";
    m_EventAdmin->subscribeSlot(this, SLOT(handleEvent(ctkEvent)), propsForSlot);
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
    this->OnPartChanged("closed");

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
    this->OnPartChanged("visible");

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
    this->OnPartChanged("hidden");

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
      QString topic = event.getProperty(ctkEventConstants::EVENT_TOPIC).toString();

      if (topic == "uk/ac/ucl/cmic/midasnavigationview/SLICE_CHANGED")
      {
        QString slice = event.getProperty("slice_number").toString();
        m_MIDASMultiViewWidget->SetSelectedWindowSliceNumber(slice.toInt());
      }
      else if (topic == "uk/ac/ucl/cmic/midasnavigationview/MAGNIFICATION_CHANGED")
      {
        QString magnification = event.getProperty("magnification_factor").toString();
        m_MIDASMultiViewWidget->SetSelectedWindowMagnification(magnification.toInt());
      }
      if (topic == "uk/ac/ucl/cmic/midasnavigationview/TIME_CHANGED")
      {
        QString time = event.getProperty("time_step").toString();
        m_MIDASMultiViewWidget->SetSelectedTimeStep(time.toInt());
      }
      else if (topic == "uk/ac/ucl/cmic/midasnavigationview/ORIENTATION_CHANGED")
      {
        QString orientation = event.getProperty("orientation").toString();
        if (orientation == "axial")
        {
          m_MIDASMultiViewWidget->SetSelectedWindowToAxial();
        }
        else if (orientation == "sagittal")
        {
          m_MIDASMultiViewWidget->SetSelectedWindowToSagittal();
        }
        else if (orientation == "coronal")
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

void QmitkMIDASMultiViewEditor::OnUpdateMIDASViewingControlsValues(UpdateMIDASViewingControlsInfo info)
{
  try
  {
    ctkDictionary properties;
    properties["min_time"] = info.minTime;
    properties["max_time"] = info.maxTime;
    properties["min_slice"] = info.minSlice;
    properties["max_slice"] = info.maxSlice;
    properties["min_magnification"] = info.minMagnification;
    properties["max_magnification"] = info.maxMagnification;
    properties["current_time"] = info.currentTime;
    properties["current_slice"] = info.currentSlice;
    properties["current_magnification"] = info.currentMagnification;
    if (info.isAxial)
    {
      properties["orientation"] = "axial";
    }
    else if (info.isSagittal)
    {
      properties["orientation"] = "sagittal";
    }
    else if (info.isCoronal)
    {
      properties["orientation"] = "coronal";
    }

    emit UpdateMIDASViewingControlsValues(properties);
  }
  catch (const ctkRuntimeException& e)
  {
    MITK_ERROR << "QmitkMIDASMultiViewEditor::OnUpdateMIDASViewingControls, failed with:" << e.what() \
        << ", caused by " << e.getCause().toLocal8Bit().constData() \
        << std::endl;
  }
}

void QmitkMIDASMultiViewEditor::OnPartChanged(QString status)
{
  try
  {
    ctkDictionary properties;
    properties["part_status"] = status;
    emit PartStatusChanged(properties);
  }
  catch (const ctkRuntimeException& e)
  {
    MITK_ERROR << "QmitkMIDASMultiViewEditor::OnPartChanged, failed with:" << e.what() \
        << ", caused by " << e.getCause().toLocal8Bit().constData() \
        << std::endl;
  }
}

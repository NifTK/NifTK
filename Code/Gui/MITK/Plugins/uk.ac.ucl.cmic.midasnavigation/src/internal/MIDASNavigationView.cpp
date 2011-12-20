/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-05 18:07:46 +0000 (Mon, 05 Dec 2011) $
 Revision          : $Revision: 7922 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
 
#include "MIDASNavigationView.h"
#include "MIDASNavigationViewActivator.h"

// Qt
#include <QMessageBox>

// Blueberry
#include <berryISelectionService.h>
#include <berryIWorkbenchWindow.h>
#include <berryQtAssistantUtil.h>

// Qmitk
#include "QmitkStdMultiWidget.h"

// CTK
#include "ctkPluginContext.h"
#include "service/event/ctkEvent.h"
#include "service/event/ctkEventConstants.h"

const std::string MIDASNavigationView::VIEW_ID = "uk.ac.ucl.cmic.midasnavigationview";

MIDASNavigationView::MIDASNavigationView()
: QmitkMIDASBaseFunctionality()
, m_NavigationViewControls(NULL)
, m_Context(NULL)
, m_EventAdmin(NULL)
{
}

MIDASNavigationView::~MIDASNavigationView()
{
  if (m_NavigationViewControls != NULL)
  {
    delete m_NavigationViewControls;
  }
}

std::string MIDASNavigationView::GetViewID() const
{
  return VIEW_ID;
}

void MIDASNavigationView::CreateQtPartControl( QWidget *parent )
{
  m_Parent = parent;

  if (!m_NavigationViewControls)
  {
    m_NavigationViewControls = new MIDASNavigationViewControlsImpl();
    m_NavigationViewControls->setupUi(parent);

    QmitkMIDASBaseFunctionality::CreateQtPartControl(parent);

    connect(m_NavigationViewControls->m_AxialRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnAxialRadioButtonToggled(bool)));
    connect(m_NavigationViewControls->m_CoronalRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnCoronalRadioButtonToggled(bool)));
    connect(m_NavigationViewControls->m_SagittalRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnSagittalRadioButtonToggled(bool)));
    connect(m_NavigationViewControls->m_SliceSelectionWidget, SIGNAL(SliceNumberChanged(int, int)), this, SLOT(OnSliceNumberChanged(int, int)));
    connect(m_NavigationViewControls->m_MagnificationFactorWidget, SIGNAL(MagnificationFactorChanged(int, int)), this, SLOT(OnMagnificationFactorChanged(int, int)));

    m_Context = mitk::MIDASNavigationViewActivator::GetPluginContext();
    m_EventAdminRef = m_Context->getServiceReference<ctkEventAdmin>();
    m_EventAdmin = m_Context->getService<ctkEventAdmin>(m_EventAdminRef);

    m_EventAdmin->publishSignal(this, SIGNAL(SliceNumberChanged(ctkDictionary)),
                              "uk/ac/ucl/cmic/midasnavigationview/SLICE_CHANGED", Qt::QueuedConnection);

    m_EventAdmin->publishSignal(this, SIGNAL(MagnificationChanged(ctkDictionary)),
                              "uk/ac/ucl/cmic/midasnavigationview/MAGNIFICATION_CHANGED", Qt::QueuedConnection);

    m_EventAdmin->publishSignal(this, SIGNAL(OrientationChanged(ctkDictionary)),
                              "uk/ac/ucl/cmic/midasnavigationview/ORIENTATION_CHANGED", Qt::QueuedConnection);

    ctkDictionary propsForSlot;
    propsForSlot[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/gui/qt/common/QmitkMIDASMultiViewEditor/*";
    m_EventAdmin->subscribeSlot(this, SLOT(handleEvent(ctkEvent)), propsForSlot);
  }
}

void MIDASNavigationView::OnAxialRadioButtonToggled(bool isToggled)
{
  if (isToggled)
  {
    ctkDictionary properties;
    properties["orientation"] = "axial";
    emit OrientationChanged(properties);
  }
}

void MIDASNavigationView::OnCoronalRadioButtonToggled(bool isToggled)
{
  if (isToggled)
  {
    ctkDictionary properties;
    properties["orientation"] = "coronal";
    emit OrientationChanged(properties);
  }
}

void MIDASNavigationView::OnSagittalRadioButtonToggled(bool isToggled)
{
  if (isToggled)
  {
    ctkDictionary properties;
    properties["orientation"] = "sagittal";
    emit OrientationChanged(properties);
  }
}

void MIDASNavigationView::OnSliceNumberChanged(int oldSliceNumber, int newSliceNumber)
{
  ctkDictionary properties;
  properties["slice_number"] = newSliceNumber;
  emit SliceNumberChanged(properties);
}

void MIDASNavigationView::OnMagnificationFactorChanged(int oldMagnificationFactor, int newMagnificationFactor)
{
  ctkDictionary properties;
  properties["magnification_factor"] = newMagnificationFactor;
  emit MagnificationChanged(properties);
}

void MIDASNavigationView::SetBlockSignals(bool blockSignals)
{
  m_NavigationViewControls->m_AxialRadioButton->blockSignals(blockSignals);
  m_NavigationViewControls->m_SagittalRadioButton->blockSignals(blockSignals);
  m_NavigationViewControls->m_CoronalRadioButton->blockSignals(blockSignals);
  m_NavigationViewControls->m_SliceSelectionWidget->blockSignals(blockSignals);
  m_NavigationViewControls->m_MagnificationFactorWidget->blockSignals(blockSignals);
}

void MIDASNavigationView::handleEvent(const ctkEvent& event)
{
  try
  {
    // Ultra-cautious... block all signals.
    this->SetBlockSignals(true);

    QString topic = event.getProperty(ctkEventConstants::EVENT_TOPIC).toString();
    if (topic == "uk/ac/ucl/cmic/gui/qt/common/QmitkMIDASMultiViewEditor/OnUpdateMIDASViewingControlsValues")
    {
      QString orientation = event.getProperty("orientation").toString();
      if (orientation == "axial")
      {
        m_NavigationViewControls->m_AxialRadioButton->setChecked(true);
      }
      if (orientation == "sagittal")
      {
        m_NavigationViewControls->m_SagittalRadioButton->setChecked(true);
      }
      if (orientation == "coronal")
      {
        m_NavigationViewControls->m_CoronalRadioButton->setChecked(true);
      }
      m_NavigationViewControls->m_MagnificationFactorWidget->SetMagnificationFactor(event.getProperty("current_magnification").toInt());
      m_NavigationViewControls->m_SliceSelectionWidget->SetSliceNumber(event.getProperty("current_slice").toInt());
    }
    else
    {
      m_NavigationViewControls->m_SliceSelectionWidget->SetMinimum(event.getProperty("min_slice").toInt());
      m_NavigationViewControls->m_SliceSelectionWidget->SetMaximum(event.getProperty("max_slice").toInt());
      m_NavigationViewControls->m_MagnificationFactorWidget->SetMinimum(event.getProperty("min_magnification").toInt());
      m_NavigationViewControls->m_MagnificationFactorWidget->SetMaximum(event.getProperty("max_magnification").toInt());
    }

    // Turn signals back on again.
    this->SetBlockSignals(false);
  }
  catch (const ctkRuntimeException& e)
  {
    MITK_ERROR << "MIDASNavigationView::handleEvent, failed with:" << e.what() \
        << ", caused by " << e.getCause().toLocal8Bit().constData() \
        << std::endl;
  }
}

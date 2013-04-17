/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Qmitk
#include "SurfaceReconView.h"
#include <ctkDictionary.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include "SurfaceReconViewActivator.h"

const std::string SurfaceReconView::VIEW_ID = "uk.ac.ucl.cmic.igisurfacerecon";

//-----------------------------------------------------------------------------
SurfaceReconView::SurfaceReconView()
{
  m_SurfaceReconstruction = mitk::SurfaceReconstruction::New();
}


//-----------------------------------------------------------------------------
SurfaceReconView::~SurfaceReconView()
{
}


//-----------------------------------------------------------------------------
std::string SurfaceReconView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void SurfaceReconView::CreateQtPartControl( QWidget *parent )
{
  setupUi(parent);
  connect(DoItButton, SIGNAL(clicked()), this, SLOT(DoSurfaceReconstruction()));

  ctkServiceReference ref = mitk::SurfaceReconViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
  if (ref)
  {
    ctkEventAdmin* eventAdmin = mitk::SurfaceReconViewActivator::getContext()->getService<ctkEventAdmin>(ref);
    ctkDictionary properties;
    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
    eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
  }
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void SurfaceReconView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void SurfaceReconView::RetrievePreferenceValues()
{
  berry::IPreferences::Pointer prefs = GetPreferences();
  if (prefs.IsNotNull())
  {

  }
}


//-----------------------------------------------------------------------------
void SurfaceReconView::SetFocus()
{
}


//-----------------------------------------------------------------------------
void SurfaceReconView::OnUpdate(const ctkEvent& event)
{
  // Optional. This gets called everytime the data sources are updated.
  // If the surface reconstruction was as fast as the GUI update, we could trigger it here.

  // not sure if enum'ing the storage here is a good idea
  // FIXME: we should register a listener on the data-storage instead?
  mitk::DataStorage* storage = GetDataStorage();
  if (storage)
  {
    // leave the editable string part intact!
    // it's extremely annoying having that reset all the time while trying to input something.
    QString leftText  = LeftChannelNodeNameComboBox->currentText();
    QString rightText = RightChannelNodeNameComboBox->currentText();

    bool  wasModified = false;

    // for all elements that currently are in the combo box
    // check whether there still is a node with that name.
    for (int i = 0; i < LeftChannelNodeNameComboBox->count(); )
    {
      QString itemName = LeftChannelNodeNameComboBox->itemText(i);
      // for now, both left and right have to have the same node names.
      assert(RightChannelNodeNameComboBox->itemText(i) == itemName);

      if (storage->GetNamedNode(itemName.toStdString()) == 0)
      {
        // no node with that name (anymore)
        // so drop it from the list.
        LeftChannelNodeNameComboBox->removeItem(i);
        RightChannelNodeNameComboBox->removeItem(i);
        wasModified = true;
      }
      else
        ++i;
    }

    mitk::DataStorage::SetOfObjects::ConstPointer allNodes = storage->GetAll();
    // once here, we made sure that only the combobox has items only that exist in data-storage.
    // that means, data-storage has at least the same amount of nodes as there are combobox items.
    assert(allNodes->Size() >= LeftChannelNodeNameComboBox->count());

    if (allNodes->Size() > LeftChannelNodeNameComboBox->count())
    {
      for (mitk::DataStorage::SetOfObjects::ConstIterator i = allNodes->Begin(); i != allNodes->End(); ++i)
      {
        std::string nodeName = i->Value()->GetName();
        if (!nodeName.empty())
        {
          LeftChannelNodeNameComboBox->addItem(QString::fromStdString(nodeName));
          RightChannelNodeNameComboBox->addItem(QString::fromStdString(nodeName));
          wasModified = true;
        }
      }
    }

    // put original text in only if we modified the combobox.
    // otherwise the edit control is reset all the time.
    if (wasModified)
    {
      LeftChannelNodeNameComboBox->setEditText(leftText);
      RightChannelNodeNameComboBox->setEditText(rightText);
    }

    assert(LeftChannelNodeNameComboBox->count() == RightChannelNodeNameComboBox->count());
  }
}


//-----------------------------------------------------------------------------
void SurfaceReconView::DoSurfaceReconstruction()
{
  mitk::DataStorage* storage = this->GetDataStorage();

  // Extract images from the correct data node.

  // Then delagate everything to class outside of plugin, so we can unit test it.
  m_SurfaceReconstruction->Run(storage, NULL, NULL);
}

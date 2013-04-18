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
  UpdateNodeNameComboBox();
}


//-----------------------------------------------------------------------------
void SurfaceReconView::UpdateNodeNameComboBox()
{
  mitk::DataStorage::Pointer storage = GetDataStorage();
  if (storage.IsNotNull())
  {
    // leave the editable string part intact!
    // it's extremely annoying having that reset all the time while trying to input something.
    QString leftText  = LeftChannelNodeNameComboBox->currentText();
    QString rightText = RightChannelNodeNameComboBox->currentText();

    bool  wasModified = false;

    std::set<std::string>   nodeNamesLeftToAdd;

    mitk::DataStorage::SetOfObjects::ConstPointer allNodes = storage->GetAll();
    for (mitk::DataStorage::SetOfObjects::ConstIterator i = allNodes->Begin(); i != allNodes->End(); ++i)
    {
      const mitk::DataNode::Pointer node = i->Value();
      assert(node.IsNotNull());

      std::string nodeName = node->GetName();
      if (!nodeName.empty())
      {
        mitk::BaseData::Pointer data = node->GetData();
        if (data.IsNotNull())
        {
          mitk::Image::Pointer imageInNode = dynamic_cast<mitk::Image*>(data.GetPointer());
          if (imageInNode.IsNotNull())
          {
            nodeNamesLeftToAdd.insert(nodeName);
          }
        }
      }
    }

    // for all elements that currently are in the combo box
    // check whether there still is a node with that name.
    for (int i = 0; i < LeftChannelNodeNameComboBox->count(); ++i)
    {
      QString itemName = LeftChannelNodeNameComboBox->itemText(i);
      // for now, both left and right have to have the same node names.
      assert(RightChannelNodeNameComboBox->itemText(i) == itemName);

      std::set<std::string>::iterator ni = nodeNamesLeftToAdd.find(itemName.toStdString());
      if (ni == nodeNamesLeftToAdd.end())
      {
        // the node name currently in the combobox is not in data storage
        // so we need to drop it from the combobox
        LeftChannelNodeNameComboBox->removeItem(i);
        RightChannelNodeNameComboBox->removeItem(i);
        wasModified = true;
      }
      else
      {
        // name is still in data-storage
        // so remove it from the to-be-added list
        nodeNamesLeftToAdd.erase(ni);
      }
    }

    for (std::set<std::string>::const_iterator i = nodeNamesLeftToAdd.begin(); i != nodeNamesLeftToAdd.end(); ++i)
    {
      QString s = QString::fromStdString(*i);
      LeftChannelNodeNameComboBox->addItem(s);
      RightChannelNodeNameComboBox->addItem(s);
      wasModified = true;
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
  mitk::DataStorage::Pointer storage = GetDataStorage();
  if (storage.IsNotNull())
  {
    std::string leftText  = LeftChannelNodeNameComboBox->currentText().toStdString();
    std::string rightText = RightChannelNodeNameComboBox->currentText().toStdString();

    const mitk::DataNode::Pointer leftNode  = storage->GetNamedNode(leftText);
    const mitk::DataNode::Pointer rightNode = storage->GetNamedNode(rightText);

    if (leftNode.IsNotNull() && rightNode.IsNotNull())
    {
      mitk::BaseData::Pointer leftData  = leftNode->GetData();
      mitk::BaseData::Pointer rightData = rightNode->GetData();

      if (leftData.IsNotNull() && rightData.IsNotNull())
      {
        mitk::Image::Pointer leftImage  = dynamic_cast<mitk::Image*>(leftData.GetPointer());
        mitk::Image::Pointer rightImage = dynamic_cast<mitk::Image*>(rightData.GetPointer());

        if (leftImage.IsNotNull() && rightImage.IsNotNull())
        {
          try
          {
            // Then delagate everything to class outside of plugin, so we can unit test it.
            m_SurfaceReconstruction->Run(storage, leftImage, rightImage);
          }
          catch (const std::exception& e)
          {
            std::cerr << "Whoops... something went wrong with surface reconstruction: " << e.what() << std::endl;
            // FIXME: show an error message on the plugin panel somewhere?
          }
        }
      }
    }
  }
}

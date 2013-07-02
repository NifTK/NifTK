/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <berryISelectionService.h>
#include <berryIWorkbenchWindow.h>
#include <berryIPreferencesService.h>
#include <berryIBerryPreferences.h>
#include "UndistortView.h"
#include "UndistortViewPreferencesPage.h"
#include "uk_ac_ucl_cmic_igiundistort_Activator.h"
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include <QMessageBox>
#include <QTableWidgetItem>
#include <QFileDialog>
#include <Undistortion.h>
#include <boost/typeof/typeof.hpp>


const std::string UndistortView::VIEW_ID = "uk.ac.ucl.cmic.igiundistort";


//-----------------------------------------------------------------------------
UndistortView::UndistortView()
{
}


//-----------------------------------------------------------------------------
UndistortView::~UndistortView()
{
  bool  ok = false;
  ok = disconnect(m_NodeTable, SIGNAL(cellDoubleClicked(int, int)), this, SLOT(OnCellDoubleClicked(int, int)));
  assert(ok);
  ok = disconnect(m_DoItNowButton, SIGNAL(clicked()), this, SLOT(OnGoButtonClick()));
  assert(ok);
  ok = disconnect(this, SIGNAL(SignalDeferredNodeTableUpdate()), this, SLOT(OnDeferredNodeTableUpdate()));
  assert(ok);

  mitk::DataStorage::Pointer storage = GetDataStorage();
  if (storage.IsNotNull())
  {
    storage->AddNodeEvent.RemoveListener(mitk::MessageDelegate1<UndistortView, const mitk::DataNode*>(this, &UndistortView::DataStorageEventListener));
    storage->RemoveNodeEvent.RemoveListener(mitk::MessageDelegate1<UndistortView, const mitk::DataNode*>(this, &UndistortView::DataStorageEventListener));
  }
}


//-----------------------------------------------------------------------------
void UndistortView::OnUpdate(const ctkEvent& event)
{
  if (m_AutomaticUpdateRadioButton->isChecked())
  {
    OnGoButtonClick();
  }
}


//-----------------------------------------------------------------------------
void UndistortView::UpdateNodeTable()
{
  mitk::DataStorage::Pointer storage = GetDataStorage();
  if (storage.IsNotNull())
  {
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
        // only list nodes that are not output of our own undistortion
        bool hasBeenCorrected = false;
        mitk::BaseProperty::Pointer cbp = node->GetProperty(niftk::Undistortion::s_ImageIsUndistortedPropertyName);
        if (cbp.IsNotNull())
        {
          mitk::BoolProperty::Pointer c = dynamic_cast<mitk::BoolProperty*>(cbp.GetPointer());
          if (c.IsNotNull())
          {
            hasBeenCorrected = c->GetValue();
          }
        }

        if (!hasBeenCorrected)
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
    }

    for (int i = 0; i < m_NodeTable->rowCount(); )
    {
      QTableWidgetItem* item = m_NodeTable->item(i, 0);
      if (item != 0)
      {
        std::string itemName = item->text().toStdString();

        std::set<std::string>::iterator ni = nodeNamesLeftToAdd.find(itemName);
        if (ni == nodeNamesLeftToAdd.end())
        {
          // drop it off table.
          // note: removing the row will change the index for the next element!
          // that's why there is no ++i in the top for statement.
          m_NodeTable->removeRow(i);
          // drop if off the cache
          BOOST_AUTO(ci, m_UndistortionMap.find(itemName));
          if (ci != m_UndistortionMap.end())
          {
            m_UndistortionMap.erase(ci);
          }

          wasModified = true;
        }
        else
        {
          // name is still in data-storage
          // so remove it from the to-be-added list
          nodeNamesLeftToAdd.erase(ni);
          // check next row in the table.
          ++i;
        }
      }
    }

    for (std::set<std::string>::const_iterator i = nodeNamesLeftToAdd.begin(); i != nodeNamesLeftToAdd.end(); ++i)
    {
      QString s = QString::fromStdString(*i);
      int j = m_NodeTable->rowCount();
      m_NodeTable->insertRow(j);

      QTableWidgetItem* nodenameitem = new QTableWidgetItem;
      nodenameitem->setText(s);
      nodenameitem->setCheckState(Qt::Unchecked);
      nodenameitem->setFlags(Qt::ItemIsEnabled | Qt::ItemIsUserCheckable | Qt::ItemIsEnabled | Qt::ItemIsSelectable);
      m_NodeTable->setItem(j, 0, nodenameitem);

      QTableWidgetItem* outputitem = new QTableWidgetItem;
      outputitem->setText(s + "-undistorted");
      outputitem->setFlags(Qt::ItemIsEditable | Qt::ItemIsEnabled | Qt::ItemIsSelectable);
      m_NodeTable->setItem(j, 2, outputitem);

      wasModified = true;
    }

    // this is very annoying
    //m_NodeTable->resizeColumnsToContents();
    //m_NodeTable->resizeRowsToContents();
  }
}


//-----------------------------------------------------------------------------
void UndistortView::OnGoButtonClick()
{
  mitk::DataStorage::Pointer storage = GetDataStorage();

  for (int i = 0; i < m_NodeTable->rowCount(); ++i)
  {
    QTableWidgetItem* nameitem = m_NodeTable->item(i, 0);
    // we should always have an item in the node-name column...
    assert(nameitem != 0);

    QTableWidgetItem* calibparamitem = m_NodeTable->item(i, 1);

    QTableWidgetItem* outputitem = m_NodeTable->item(i, 2);
    // if there's no name for the output node then don't do anything
    assert(outputitem != 0);
    if (outputitem->text().isEmpty())
    {
      nameitem->setCheckState(Qt::Unchecked);
    }

    if (nameitem->checkState() == Qt::Checked)
    {
      // ...but the name might be empty
      std::string   nodename = nameitem->text().toStdString();
      if (!nodename.empty())
      {
        mitk::DataNode::Pointer   inputNode = storage->GetNamedNode(nodename);
        // FIXME: is this a hard error? i'm a bit undecided...
        assert(inputNode.IsNotNull());

        // as long as we have an Undistortion object it will take care of validating itself
        BOOST_AUTO(ci, m_UndistortionMap.find(nodename));
        if (ci == m_UndistortionMap.end())
        {
          niftk::Undistortion*  undist = new niftk::Undistortion(inputNode);
          // store the new Undistortion object in our cache
          // note: insert() returns an iterator to the new insertion location
          ci = m_UndistortionMap.insert(std::make_pair(nodename, undist)).first;
        }

        mitk::DataNode::Pointer   outputNode = storage->GetNamedNode(outputitem->text().toStdString());
        if (outputNode.IsNull())
        {
          // Undistortion takes care of sizing the output image correcly, etc
          // we just need to make sure the output node exists
          outputNode = mitk::DataNode::New();
          outputNode->SetName(outputitem->text().toStdString());
        }

        // FIXME: this is not a good place to load/override the node's calibration. put it somewhere else.
        if (calibparamitem != 0)
        {
          niftk::Undistortion::LoadIntrinsicCalibration(calibparamitem->text().toStdString(), inputNode);
        }

        ci->second->Run(outputNode);

        if (storage->GetNamedNode(outputitem->text().toStdString()) == NULL)
        {
          storage->Add(outputNode, inputNode);
        }
      }
      else
      {
        std::cerr << "Undistortion: skipping node with empty name." << std::endl;
      }
    }
  }
}


//-----------------------------------------------------------------------------
void UndistortView::OnCellDoubleClicked(int row, int column)
{
  QTableWidgetItem* nameitem = m_NodeTable->item(row, 0);
  if (nameitem != 0)
  {
    QString   nodename = nameitem->text();

    switch (column)
    {
      // clicked on name
      case 0:
        break;
      // clicked on param
      case 1:
      {
        QString   file = QFileDialog::getOpenFileName(GetParent(), "Intrinsic Camera Calibration", m_LastFile);
        if (!file.isEmpty())
        {
          QTableWidgetItem*   filenameitem = new QTableWidgetItem;
          filenameitem->setText(file);
          filenameitem->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
          m_NodeTable->setItem(row, column, filenameitem);

          m_LastFile = file;
        }
        break;
      }
      // clicked on output
      case 2:
        break;
    }
  }
}


//-----------------------------------------------------------------------------
void UndistortView::DataStorageEventListener(const mitk::DataNode* node)
{
  // this callback is called before the node has been removed.
  // so lets queue a signal-slot callback for later.
  emit SignalDeferredNodeTableUpdate();
}


//-----------------------------------------------------------------------------
void UndistortView::OnDeferredNodeTableUpdate()
{
  UpdateNodeTable();
}


//-----------------------------------------------------------------------------
void UndistortView::CreateQtPartControl(QWidget* parent)
{
  setupUi(parent);
  m_NodeTable->clearContents();
  // refit the columns. there's no built-in easy way for this.
  // ah bugger: this doesnt work, columns are squashed
  //for (int i = 0; i < m_NodeTable->columnCount(); ++i)
  //{
  //  m_NodeTable->setColumnWidth(i, m_NodeTable->width() / m_NodeTable->columnCount());
  //}

  bool  ok = false;
  ok = connect(m_NodeTable, SIGNAL(cellDoubleClicked(int, int)), this, SLOT(OnCellDoubleClicked(int, int)));
  assert(ok);
  ok = connect(m_DoItNowButton, SIGNAL(clicked()), this, SLOT(OnGoButtonClick()));
  assert(ok);
  ok = connect(this, SIGNAL(SignalDeferredNodeTableUpdate()), this, SLOT(OnDeferredNodeTableUpdate()), Qt::QueuedConnection);
  assert(ok);

  RetrievePreferenceValues();

  // this thing fires off events when the data inside the node has been modified
  ctkServiceReference ref = mitk::uk_ac_ucl_cmic_igiundistort_Activator::getContext()->getServiceReference<ctkEventAdmin>();
  if (ref)
  {
    ctkEventAdmin* eventAdmin = mitk::uk_ac_ucl_cmic_igiundistort_Activator::getContext()->getService<ctkEventAdmin>(ref);
    ctkDictionary properties;
    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
    eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
  }

  mitk::DataStorage::Pointer storage = GetDataStorage();
  if (storage.IsNotNull())
  {
    storage->AddNodeEvent.AddListener(mitk::MessageDelegate1<UndistortView, const mitk::DataNode*>(this, &UndistortView::DataStorageEventListener));
    storage->RemoveNodeEvent.AddListener(mitk::MessageDelegate1<UndistortView, const mitk::DataNode*>(this, &UndistortView::DataStorageEventListener));
  }

  UpdateNodeTable();
}


//-----------------------------------------------------------------------------
void UndistortView::SetFocus()
{
}


//-----------------------------------------------------------------------------
void UndistortView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  // Retrieve up-to-date preference values.
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void UndistortView::RetrievePreferenceValues()
{
  berry::IPreferencesService::Pointer prefService = berry::Platform::GetServiceRegistry().GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);
  berry::IBerryPreferences::Pointer prefs = (prefService->GetSystemPreferences()->Node(UndistortViewPreferencesPage::s_PrefsNodeName)).Cast<berry::IBerryPreferences>();
  assert(prefs);

  m_LastFile = QString::fromStdString(prefs->Get(UndistortViewPreferencesPage::s_DefaultCalibrationFilePathPrefsName, ""));
}

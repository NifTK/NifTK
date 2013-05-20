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
#include "uk_ac_ucl_cmic_undistort_Activator.h"
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include <QMessageBox>
#include <QTableWidgetItem>


const std::string UndistortView::VIEW_ID = "uk.ac.ucl.cmic.undistort";


//-----------------------------------------------------------------------------
UndistortView::UndistortView()
{
}


//-----------------------------------------------------------------------------
UndistortView::~UndistortView()
{
}


//-----------------------------------------------------------------------------
void UndistortView::OnUpdate(const ctkEvent& event)
{
  UpdateNodeTable();
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

    for (int i = 0; i < m_NodeTable->rowCount(); ++i)
    {
      QTableWidgetItem* item = m_NodeTable->item(i, 0);
      if (item != 0)
      {
        QString itemName = item->text();

        std::set<std::string>::iterator ni = nodeNamesLeftToAdd.find(itemName.toStdString());
        if (ni == nodeNamesLeftToAdd.end())
        {
          m_NodeTable->removeRow(i);
          wasModified = true;
        }
        else
        {
          // name is still in data-storage
          // so remove it from the to-be-added list
          nodeNamesLeftToAdd.erase(ni);
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
      outputitem->setText(s + "-corrected");
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
void UndistortView::CreateQtPartControl(QWidget* parent)
{
  setupUi(parent);
  m_NodeTable->clear();

  // void QTableWidget::setCellWidget ( int row, int column, QWidget * widget )

  RetrievePreferenceValues();


  ctkServiceReference ref = mitk::uk_ac_ucl_cmic_undistort_Activator::getContext()->getServiceReference<ctkEventAdmin>();
  if (ref)
  {
    ctkEventAdmin* eventAdmin = mitk::uk_ac_ucl_cmic_undistort_Activator::getContext()->getService<ctkEventAdmin>(ref);
    ctkDictionary properties;
    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
    eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
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
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  berry::IBerryPreferences::Pointer prefs
      = (prefService->GetSystemPreferences()->Node(VIEW_ID))
        .Cast<berry::IBerryPreferences>();
  assert( prefs );

#if 0
  m_AutoUpdate = prefs->GetBool(ImageStatisticsViewPreferencesPage::AUTO_UPDATE_NAME, false);
  m_AssumeBinary = prefs->GetBool(ImageStatisticsViewPreferencesPage::ASSUME_BINARY_NAME, true);
  m_RequireSameSizeImage = prefs->GetBool(ImageStatisticsViewPreferencesPage::REQUIRE_SAME_SIZE_IMAGE_NAME, true);
  m_BackgroundValue = prefs->GetInt(ImageStatisticsViewPreferencesPage::BACKGROUND_VALUE_NAME, 0);
#endif
}

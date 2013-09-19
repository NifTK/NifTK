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
#include <QtConcurrentRun>
#include <mitkCameraIntrinsicsProperty.h>


const std::string UndistortView::VIEW_ID = "uk.ac.ucl.cmic.igiundistort";


//-----------------------------------------------------------------------------
UndistortView::UndistortView()
{
  bool ok = false;
  ok = connect(&m_BackgroundProcessWatcher, SIGNAL(finished()), this, SLOT(OnBackgroundProcessFinished()));
  assert(ok);
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

  // wait for it to finish first and then disconnect?
  // or the other way around?
  // i'd say disconnect first then wait because at that time we no longer care about the result
  // and the finished-handler might access some half-destroyed objects.
  // side note: what happens if background-process is running and emits a signal just before we disconnect.
  // that signal is then queued on our to-be-destructed object. does it get delivered or silently dropped?
  ok = disconnect(&m_BackgroundProcessWatcher, SIGNAL(finished()), this, SLOT(OnBackgroundProcessFinished()));
  assert(ok);
  m_BackgroundProcessWatcher.waitForFinished();
  // m_BackgroundQueue is cleared in the finished-signal-handler
  // which we just disconnected above.
  m_BackgroundQueue.clear();

  for (std::map<mitk::Image::Pointer, niftk::Undistortion*>::iterator i = m_UndistortionMap.begin(); i != m_UndistortionMap.end(); ++i)
  {
    delete i->second;
    i->second = 0;
  }
  m_UndistortionMap.clear();
}


//-----------------------------------------------------------------------------
void UndistortView::OnUpdate(const ctkEvent& event)
{
  if (m_AutomaticUpdateRadioButton->isChecked())
  {
    m_DoItNowButton->click();
  }
}


//-----------------------------------------------------------------------------
void UndistortView::UpdateNodeTable()
{
  mitk::DataStorage::Pointer storage = GetDataStorage();
  if (storage.IsNotNull())
  {
    bool  wasModified = false;

    std::set<std::string>             nodeNamesLeftToAdd;
    std::set<mitk::Image::Pointer>    cacheItemsToKeep;

    mitk::DataStorage::SetOfObjects::ConstPointer allNodes = storage->GetAll();
    for (mitk::DataStorage::SetOfObjects::ConstIterator i = allNodes->Begin(); i != allNodes->End(); ++i)
    {
      const mitk::DataNode::Pointer node = i->Value();
      assert(node.IsNotNull());

      std::string nodeName = node->GetName();
      if (!nodeName.empty())
      {
        mitk::Image::Pointer imageInNode = dynamic_cast<mitk::Image*>(node->GetData());
        if (imageInNode.IsNotNull())
        {
          // only list nodes that are not output of our own undistortion
          bool hasBeenCorrected = false;
          mitk::BoolProperty::Pointer cbp = dynamic_cast<mitk::BoolProperty*>(imageInNode->GetProperty(niftk::Undistortion::s_ImageIsUndistortedPropertyName).GetPointer());
          if (cbp.IsNull())
          {
            // image doesnt have right props, check node
            cbp = dynamic_cast<mitk::BoolProperty*>(node->GetProperty(niftk::Undistortion::s_ImageIsUndistortedPropertyName));
          }
          else
          {
            // debug: if node and image have the prop then they need to match!
            mitk::BoolProperty::Pointer n = dynamic_cast<mitk::BoolProperty*>(node->GetProperty(niftk::Undistortion::s_ImageIsUndistortedPropertyName));
            if (n.IsNotNull())
            {
              assert(n->GetValue() == cbp->GetValue());
            }
          }

          if (cbp.IsNotNull())
          {
            hasBeenCorrected = cbp->GetValue();
          }

          if (!hasBeenCorrected)
          {
            nodeNamesLeftToAdd.insert(nodeName);
            cacheItemsToKeep.insert(imageInNode);
          }
        }
      }
    }

    // go through the table and check which nodes are still there
    for (int i = 0; i < m_NodeTable->rowCount(); )
    {
      QTableWidgetItem* item = m_NodeTable->item(i, 0);
      if (item != 0)
      {
        std::string itemName = item->text().toStdString();

        // does the node in the table still exist in data storage?
        std::set<std::string>::iterator ni = nodeNamesLeftToAdd.find(itemName);
        if (ni == nodeNamesLeftToAdd.end())
        {
          // if not drop it off table.

          // note: removing the row will change the index for the next element!
          // that's why there is no ++i in the top for statement.
          m_NodeTable->removeRow(i);

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

    // clear out all cache items that reference images no longer on any suitable nodes.
    for (std::map<mitk::Image::Pointer, niftk::Undistortion*>::iterator i = m_UndistortionMap.begin(); i != m_UndistortionMap.end(); )
    {
      if (cacheItemsToKeep.find(i->first) == cacheItemsToKeep.end())
      {
        // in visual c++, erase() returns an iterator one past the deleted element.
        std::map<mitk::Image::Pointer, niftk::Undistortion*>::iterator j(i);
        ++j;
        m_UndistortionMap.erase(i);
        i = j;
      }
      else
      {
        ++i;
      }
    }
  }
}


//-----------------------------------------------------------------------------
void UndistortView::OnGoButtonClick()
{
  // dont do anything if we are currently running undistortion.
  // instead of disabling the go button we do this.
  // because the flickering button is really annoying.
  //if (m_BackgroundProcess.isRunning())
  // race condition: if we check whether background process is running then we could
  // end up in a situation where background is finished but future-watcher hasn't signaled us yet.
  if (!m_BackgroundQueue.empty())
  {
    return;
  }

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
        // is this a hard error? i'm a bit undecided...
        // it could simply be a result of a stale table, so sliently ignoring it would be an option.
        if (inputNode.IsNotNull())
        {
          mitk::Image::Pointer    inputImage = dynamic_cast<mitk::Image*>(inputNode->GetData());
          if (inputImage.IsNotNull())
          {
            bool    hascalib = false;
            if (calibparamitem != 0)
            {
              try
              {
                std::string   filename = calibparamitem->text().toStdString();
                // cache the parsed calib data, to avoid repeatedly (every frame!) reading the same stuff.
                std::map<std::string, mitk::CameraIntrinsicsProperty::Pointer>::iterator fc = m_ParamFileCache.find(filename);
                if (fc != m_ParamFileCache.end())
                {
                  // beware: mitk::CameraIntrinsicsProperty::Clone() does not clone its value!
                  mitk::CameraIntrinsicsProperty::Pointer calibprop = mitk::CameraIntrinsicsProperty::New(fc->second->GetValue()->Clone());
                  inputImage->SetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName, calibprop);
                  inputNode->SetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName, calibprop);
                }
                else
                {
                  // this will stick props on both node and its image
                  niftk::Undistortion::LoadIntrinsicCalibration(filename, inputNode);
                  mitk::CameraIntrinsicsProperty::Pointer calibprop = dynamic_cast<mitk::CameraIntrinsicsProperty*>(inputImage->GetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName).GetPointer());
                  assert(calibprop.IsNotNull());
                  // beware: mitk::CameraIntrinsicsProperty::Clone() does not clone its value!
                  m_ParamFileCache[filename] = mitk::CameraIntrinsicsProperty::New(calibprop->GetValue()->Clone());
                }

                hascalib = true;
              }
              catch (...)
              {
                std::cerr << "Failed loading calib data from file " << calibparamitem->text().toStdString() << std::endl;
                hascalib = false;
              }
            }
            else
            {
              // if there is no overriding calib file set for this image
              // then there has to be one defined on the image/node already.
              mitk::BaseProperty::Pointer   imgprop = inputImage->GetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName);
              if (imgprop.IsNull())
              {
                imgprop = inputNode->GetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName);
              }

              if (imgprop.IsNotNull())
              {
                mitk::CameraIntrinsicsProperty::Pointer imgcalib = dynamic_cast<mitk::CameraIntrinsicsProperty*>(imgprop.GetPointer());
                if (imgcalib.IsNotNull())
                {
                  hascalib = true;
                }
              }
            }

            // only do background undistortion if we are fairly certain that there is
            // calib data on the image/node. while Undistortion will handle things properly
            // when not, it will still generate an empty image, which is pretty confusing to the user.
            if (hascalib)
            {
              // as long as we have an Undistortion object it will take care of validating itself
              BOOST_AUTO(ci, m_UndistortionMap.find(inputImage));
              if (ci == m_UndistortionMap.end())
              {
                niftk::Undistortion*  undist = new niftk::Undistortion(inputImage);
                // store the new Undistortion object in our cache
                // note: insert() returns an iterator to the new insertion location
                ci = m_UndistortionMap.insert(std::make_pair(inputImage, undist)).first;
              }

              mitk::Image::Pointer      outputImage;
              mitk::DataNode::Pointer   outputNode = storage->GetNamedNode(outputitem->text().toStdString());
              if (outputNode.IsNotNull())
              {
                outputImage = dynamic_cast<mitk::Image*>(outputNode->GetData());
              }

              // check that output image is correct size.
              ci->second->PrepareOutput(outputImage);

              WorkItem    wi;
              wi.m_InputImage = inputImage;
              wi.m_OutputImage = outputImage;
              wi.m_OutputNodeName = outputitem->text().toStdString();
              wi.m_InputNodeName = nodename;
              wi.m_Proc = ci->second;

              m_BackgroundQueue.push_back(wi);
            }
            else
            {
              nameitem->setCheckState(Qt::Unchecked);
              std::cerr << "Undistortion: skipping node " << nodename << " without calib data" << std::endl;
            }
          }
        }
        else
        {
          std::cerr << "Undistortion: skipping unknown node with name: " << nodename << std::endl;
        }
      }
      else
      {
        std::cerr << "Undistortion: skipping node with empty name." << std::endl;
      }
    }
  }

  if (!m_BackgroundQueue.empty())
  {
    m_BackgroundProcess = QtConcurrent::run(this, &UndistortView::RunBackgroundProcessing);
    m_BackgroundProcessWatcher.setFuture(m_BackgroundProcess);
  }
}


//-----------------------------------------------------------------------------
void UndistortView::RunBackgroundProcessing()
{
  assert(!m_BackgroundQueue.empty());
  for (std::size_t i = 0; i < m_BackgroundQueue.size(); ++i)
  {
    try
    {
      m_BackgroundQueue[i].m_Proc->Run(m_BackgroundQueue[i].m_OutputImage);
    }
    catch (const std::exception& e)
    {
      std::cerr << "Caught exception while undistorting: " << e.what() << std::endl;
    }
    catch (...)
    {
      std::cerr << "Caught unknown exception while undistorting!" << std::endl;
    }
  }
}


//-----------------------------------------------------------------------------
void UndistortView::OnBackgroundProcessFinished()
{
  mitk::DataStorage::Pointer storage = GetDataStorage();
  assert(storage.IsNotNull());

  for (std::size_t i = 0; i < m_BackgroundQueue.size(); ++i)
  {
    bool  nodeIsNew = false;
    mitk::DataNode::Pointer   outputNode = storage->GetNamedNode(m_BackgroundQueue[i].m_OutputNodeName);
    if (outputNode.IsNull())
    {
      nodeIsNew = true;
      outputNode = mitk::DataNode::New();
      outputNode->SetName(m_BackgroundQueue[i].m_OutputNodeName);
    }

    outputNode->SetData(m_BackgroundQueue[i].m_OutputImage);

    if (nodeIsNew)
    {
      mitk::DataNode::Pointer   inputNode = storage->GetNamedNode(m_BackgroundQueue[i].m_InputNodeName);
      if (inputNode.IsNotNull())
      {
        storage->Add(outputNode, inputNode);
      }
      else
      {
        storage->Add(outputNode);
      }
    }

    // we'll have props on the image only, so copy them to node too.
    outputNode->SetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName,  m_BackgroundQueue[i].m_OutputImage->GetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName));
    outputNode->SetProperty(niftk::Undistortion::s_ImageIsUndistortedPropertyName, m_BackgroundQueue[i].m_OutputImage->GetProperty(niftk::Undistortion::s_ImageIsUndistortedPropertyName));
    outputNode->Modified();
  }

  m_BackgroundQueue.clear();
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
        // user chooses (or cancels) a new file, clear the cache to avoid stale data. 
        m_ParamFileCache.clear();
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

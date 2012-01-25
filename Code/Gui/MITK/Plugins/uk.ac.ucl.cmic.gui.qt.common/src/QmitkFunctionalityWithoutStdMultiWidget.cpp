/*=========================================================================

 Program:   Medical Imaging & Interaction Toolkit
 Language:  C++
 Date:      $Date$
 Version:   $Revision$

 Copyright (c) German Cancer Research Center, Division of Medical and
 Biological Informatics. All rights reserved.
 See MITKCopyright.txt or http://www.mitk.org/copyright.html for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 =========================================================================*/

#include "QmitkFunctionalityWithoutStdMultiWidget.h"
#include "internal/QmitkFunctionalityUtil.h"

// other includes
#include <mitkLogMacros.h>

// mitk Includes
#include <mitkIDataStorageService.h>
#include <mitkDataStorageEditorInput.h>

// berry Includes
#include <berryIWorkbenchPage.h>
#include <berryIBerryPreferences.h>
#include <berryIEditorPart.h>

// Qmitk Includes
#include <QmitkStdMultiWidgetEditor.h>

// Qt Includes
#include <QMessageBox>
#include <QScrollArea>
#include <QVBoxLayout>

#include <berryIWorkbenchWindow.h>
#include <berryISelectionService.h>

#include <mitkDataNodeObject.h>

QmitkFunctionalityWithoutStdMultiWidget::QmitkFunctionalityWithoutStdMultiWidget()
 : m_Parent(0)
 , m_Active(false)
 , m_Visible(false)
 , m_SelectionProvider(0)
 , m_HandlesMultipleDataStorages(false)
 , m_InDataStorageChanged(false)
{
  m_PreferencesService =
    berry::Platform::GetServiceRegistry().GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);
}

void QmitkFunctionalityWithoutStdMultiWidget::SetHandleMultipleDataStorages(bool multiple)
{
  m_HandlesMultipleDataStorages = multiple;
}

bool QmitkFunctionalityWithoutStdMultiWidget::HandlesMultipleDataStorages() const
{
  return m_HandlesMultipleDataStorages;
}

mitk::DataStorage::Pointer
QmitkFunctionalityWithoutStdMultiWidget::GetDataStorage() const
{
  mitk::IDataStorageService::Pointer service =
    berry::Platform::GetServiceRegistry().GetServiceById<mitk::IDataStorageService>(mitk::IDataStorageService::ID);

  if (service.IsNotNull())
  {
    if (m_HandlesMultipleDataStorages)
      return service->GetActiveDataStorage()->GetDataStorage();
    else
      return service->GetDefaultDataStorage()->GetDataStorage();
  }

  return 0;
}

mitk::DataStorage::Pointer QmitkFunctionalityWithoutStdMultiWidget::GetDefaultDataStorage() const
{
  mitk::IDataStorageService::Pointer service =
    berry::Platform::GetServiceRegistry().GetServiceById<mitk::IDataStorageService>(mitk::IDataStorageService::ID);

  return service->GetDefaultDataStorage()->GetDataStorage();
}

void QmitkFunctionalityWithoutStdMultiWidget::CreatePartControl(void* parent)
{

  // scrollArea
  QScrollArea* scrollArea = new QScrollArea;
  //QVBoxLayout* scrollAreaLayout = new QVBoxLayout(scrollArea);
  scrollArea->setFrameShadow(QFrame::Plain);
  scrollArea->setFrameShape(QFrame::NoFrame);
  scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

  // m_Parent
  m_Parent = new QWidget;
  //m_Parent->setSizePolicy(QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding));
  this->CreateQtPartControl(m_Parent);

  //scrollAreaLayout->addWidget(m_Parent);
  //scrollArea->setLayout(scrollAreaLayout);

  // set the widget now
  scrollArea->setWidgetResizable(true);
  scrollArea->setWidget(m_Parent);

  // add the scroll area to the real parent (the view tabbar)
  QWidget* parentQWidget = static_cast<QWidget*>(parent);
  QVBoxLayout* parentLayout = new QVBoxLayout(parentQWidget);
  parentLayout->setMargin(0);
  parentLayout->setSpacing(0);
  parentLayout->addWidget(scrollArea);

  // finally set the layout containing the scroll area to the parent widget (= show it)
  parentQWidget->setLayout(parentLayout);

  this->AfterCreateQtPartControl();
}

void QmitkFunctionalityWithoutStdMultiWidget::AfterCreateQtPartControl()
{
  // REGISTER DATASTORAGE LISTENER
  this->GetDefaultDataStorage()->AddNodeEvent.AddListener( mitk::MessageDelegate1<QmitkFunctionalityWithoutStdMultiWidget, const mitk::DataNode*>
    ( this, &QmitkFunctionalityWithoutStdMultiWidget::NodeAddedProxy ) );
  this->GetDefaultDataStorage()->ChangedNodeEvent.AddListener( mitk::MessageDelegate1<QmitkFunctionalityWithoutStdMultiWidget, const mitk::DataNode*>
    ( this, &QmitkFunctionalityWithoutStdMultiWidget::NodeChangedProxy ) );
  this->GetDefaultDataStorage()->RemoveNodeEvent.AddListener( mitk::MessageDelegate1<QmitkFunctionalityWithoutStdMultiWidget, const mitk::DataNode*>
    ( this, &QmitkFunctionalityWithoutStdMultiWidget::NodeRemovedProxy ) );

  // REGISTER PREFERENCES LISTENER
  berry::IBerryPreferences::Pointer prefs = this->GetPreferences().Cast<berry::IBerryPreferences>();
  if(prefs.IsNotNull())
    prefs->OnChanged.AddListener(berry::MessageDelegate1<QmitkFunctionalityWithoutStdMultiWidget
    , const berry::IBerryPreferences*>(this, &QmitkFunctionalityWithoutStdMultiWidget::OnPreferencesChanged));

  // REGISTER FOR WORKBENCH SELECTION EVENTS
  m_BlueBerrySelectionListener = berry::ISelectionListener::Pointer(new berry::SelectionChangedAdapter<QmitkFunctionalityWithoutStdMultiWidget>(this
    , &QmitkFunctionalityWithoutStdMultiWidget::BlueBerrySelectionChanged));
  this->GetSite()->GetWorkbenchWindow()->GetSelectionService()->AddPostSelectionListener(/*"org.mitk.views.datamanager",*/ m_BlueBerrySelectionListener);

  // REGISTER A SELECTION PROVIDER
  QmitkFunctionalitySelectionProvider::Pointer _SelectionProvider
    = QmitkFunctionalitySelectionProvider::New(this);
  m_SelectionProvider = _SelectionProvider.GetPointer();
  this->GetSite()->SetSelectionProvider(berry::ISelectionProvider::Pointer(m_SelectionProvider));

  // EMULATE INITIAL SELECTION EVENTS

  // send datamanager selection
  this->OnSelectionChanged(this->GetDataManagerSelection());

  // send preferences changed event
  this->OnPreferencesChanged(this->GetPreferences().Cast<berry::IBerryPreferences>().GetPointer());
}

void QmitkFunctionalityWithoutStdMultiWidget::ClosePart()
{

}

void QmitkFunctionalityWithoutStdMultiWidget::ClosePartProxy()
{
  this->GetDefaultDataStorage()->AddNodeEvent.RemoveListener( mitk::MessageDelegate1<QmitkFunctionalityWithoutStdMultiWidget, const mitk::DataNode*>
    ( this, &QmitkFunctionalityWithoutStdMultiWidget::NodeAddedProxy ) );
  this->GetDefaultDataStorage()->RemoveNodeEvent.RemoveListener( mitk::MessageDelegate1<QmitkFunctionalityWithoutStdMultiWidget, const mitk::DataNode*>
    ( this, &QmitkFunctionalityWithoutStdMultiWidget::NodeRemovedProxy) );
  this->GetDefaultDataStorage()->ChangedNodeEvent.RemoveListener( mitk::MessageDelegate1<QmitkFunctionalityWithoutStdMultiWidget, const mitk::DataNode*>
    ( this, &QmitkFunctionalityWithoutStdMultiWidget::NodeChangedProxy ) );

  berry::IBerryPreferences::Pointer prefs = this->GetPreferences().Cast<berry::IBerryPreferences>();
  if(prefs.IsNotNull())
  {
    prefs->OnChanged.RemoveListener(berry::MessageDelegate1<QmitkFunctionalityWithoutStdMultiWidget
    , const berry::IBerryPreferences*>(this, &QmitkFunctionalityWithoutStdMultiWidget::OnPreferencesChanged));
    // flush the preferences here (disabled, everyone should flush them by themselves at the right moment)
    // prefs->Flush();
  }

  // REMOVE SELECTION PROVIDER
  this->GetSite()->SetSelectionProvider(berry::ISelectionProvider::Pointer(NULL));

  berry::ISelectionService* s = GetSite()->GetWorkbenchWindow()->GetSelectionService();
  if(s)
  {
    s->RemovePostSelectionListener(m_BlueBerrySelectionListener);
  }

    this->ClosePart();
}

QmitkFunctionalityWithoutStdMultiWidget::~QmitkFunctionalityWithoutStdMultiWidget()
{
  this->Register();
  this->ClosePartProxy();

  this->UnRegister(false);
}

void QmitkFunctionalityWithoutStdMultiWidget::OnPreferencesChanged( const berry::IBerryPreferences* )
{
}

void QmitkFunctionalityWithoutStdMultiWidget::BlueBerrySelectionChanged(berry::IWorkbenchPart::Pointer sourcepart, berry::ISelection::ConstPointer selection)
{
  if(sourcepart.IsNull() || sourcepart->GetSite()->GetId() != "org.mitk.views.datamanager")
    return;

  mitk::DataNodeSelection::ConstPointer _DataNodeSelection
    = selection.Cast<const mitk::DataNodeSelection>();
  this->OnSelectionChanged(this->DataNodeSelectionToVector(_DataNodeSelection));
}

bool QmitkFunctionalityWithoutStdMultiWidget::IsVisible() const
{
  return m_Visible;
}

void QmitkFunctionalityWithoutStdMultiWidget::SetFocus()
{
}

void QmitkFunctionalityWithoutStdMultiWidget::Activated()
{
}

void QmitkFunctionalityWithoutStdMultiWidget::Deactivated()
{
}

void QmitkFunctionalityWithoutStdMultiWidget::DataStorageChanged()
{

}

void QmitkFunctionalityWithoutStdMultiWidget::HandleException( const char* str, QWidget* parent, bool showDialog ) const
{
  //itkGenericOutputMacro( << "Exception caught: " << str );
  MITK_ERROR << str;
  if ( showDialog )
  {
    QMessageBox::critical ( parent, "Exception caught!", str );
  }
}

void QmitkFunctionalityWithoutStdMultiWidget::HandleException( std::exception& e, QWidget* parent, bool showDialog ) const
{
  HandleException( e.what(), parent, showDialog );
}

void QmitkFunctionalityWithoutStdMultiWidget::WaitCursorOn()
{
  QApplication::setOverrideCursor( QCursor(Qt::WaitCursor) );
}

void QmitkFunctionalityWithoutStdMultiWidget::BusyCursorOn()
{
  QApplication::setOverrideCursor( QCursor(Qt::BusyCursor) );
}

void QmitkFunctionalityWithoutStdMultiWidget::WaitCursorOff()
{
  this->RestoreOverrideCursor();
}

void QmitkFunctionalityWithoutStdMultiWidget::BusyCursorOff()
{
  this->RestoreOverrideCursor();
}

void QmitkFunctionalityWithoutStdMultiWidget::RestoreOverrideCursor()
{
  QApplication::restoreOverrideCursor();
}

berry::IPreferences::Pointer QmitkFunctionalityWithoutStdMultiWidget::GetPreferences() const
{
  berry::IPreferencesService::Pointer prefService = m_PreferencesService.Lock();
  // const_cast workaround for bad programming: const uncorrectness this->GetViewSite() should be const
  std::string id = "/" + (const_cast<QmitkFunctionalityWithoutStdMultiWidget*>(this))->GetViewSite()->GetId();
  return prefService.IsNotNull() ? prefService->GetSystemPreferences()->Node(id): berry::IPreferences::Pointer(0);
}

void QmitkFunctionalityWithoutStdMultiWidget::Visible()
{

}

void QmitkFunctionalityWithoutStdMultiWidget::Hidden()
{

}

bool QmitkFunctionalityWithoutStdMultiWidget::IsExclusiveFunctionality() const
{
  return true;
}

void QmitkFunctionalityWithoutStdMultiWidget::SetVisible( bool visible )
{
  m_Visible = visible;
}

void QmitkFunctionalityWithoutStdMultiWidget::SetActivated( bool activated )
{
  m_Active = activated;
}

bool QmitkFunctionalityWithoutStdMultiWidget::IsActivated() const
{
  return m_Active;
}

std::vector<mitk::DataNode*> QmitkFunctionalityWithoutStdMultiWidget::GetCurrentSelection() const
{
  berry::ISelection::ConstPointer selection( this->GetSite()->GetWorkbenchWindow()->GetSelectionService()->GetSelection());
  // buffer for the data manager selection
  mitk::DataNodeSelection::ConstPointer currentSelection = selection.Cast<const mitk::DataNodeSelection>();
  return this->DataNodeSelectionToVector(currentSelection);
}

std::vector<mitk::DataNode*> QmitkFunctionalityWithoutStdMultiWidget::GetDataManagerSelection() const
{
  berry::ISelection::ConstPointer selection( this->GetSite()->GetWorkbenchWindow()->GetSelectionService()->GetSelection("org.mitk.views.datamanager"));
    // buffer for the data manager selection
  mitk::DataNodeSelection::ConstPointer currentSelection = selection.Cast<const mitk::DataNodeSelection>();
  return this->DataNodeSelectionToVector(currentSelection);
}

void QmitkFunctionalityWithoutStdMultiWidget::OnSelectionChanged(std::vector<mitk::DataNode*> /*nodes*/)
{
}

std::vector<mitk::DataNode*> QmitkFunctionalityWithoutStdMultiWidget::DataNodeSelectionToVector(mitk::DataNodeSelection::ConstPointer currentSelection) const
{

  std::vector<mitk::DataNode*> selectedNodes;
  if(currentSelection.IsNull())
    return selectedNodes;

  mitk::DataNodeObject* _DataNodeObject = 0;
  mitk::DataNode* _DataNode = 0;

  for(mitk::DataNodeSelection::iterator it = currentSelection->Begin();
    it != currentSelection->End(); ++it)
  {
    _DataNodeObject = dynamic_cast<mitk::DataNodeObject*>((*it).GetPointer());
    if(_DataNodeObject)
    {
      _DataNode = _DataNodeObject->GetDataNode();
      if(_DataNode)
        selectedNodes.push_back(_DataNode);
    }
  }

  return selectedNodes;
}

void QmitkFunctionalityWithoutStdMultiWidget::NodeAddedProxy( const mitk::DataNode* node )
{
  // garantuee no recursions when a new node event is thrown in NodeAdded()
  if(!m_InDataStorageChanged)
  {
    m_InDataStorageChanged = true;
    this->NodeAdded(node);
    this->DataStorageChanged();
    m_InDataStorageChanged = false;
  }

}

void QmitkFunctionalityWithoutStdMultiWidget::NodeAdded( const mitk::DataNode*  /*node*/ )
{

}

void QmitkFunctionalityWithoutStdMultiWidget::NodeRemovedProxy( const mitk::DataNode* node )
{
  // garantuee no recursions when a new node event is thrown in NodeAdded()
  if(!m_InDataStorageChanged)
  {
    m_InDataStorageChanged = true;
    this->NodeRemoved(node);
    this->DataStorageChanged();
    m_InDataStorageChanged = false;
  }
}

void QmitkFunctionalityWithoutStdMultiWidget::NodeRemoved( const mitk::DataNode*  /*node*/ )
{

}

void QmitkFunctionalityWithoutStdMultiWidget::NodeChanged( const mitk::DataNode* /*node*/ )
{

}

void QmitkFunctionalityWithoutStdMultiWidget::NodeChangedProxy( const mitk::DataNode* node )
{
  // garantuee no recursions when a new node event is thrown in NodeAdded()
  if(!m_InDataStorageChanged)
  {
    m_InDataStorageChanged = true;
    this->NodeChanged(node);
    this->DataStorageChanged();
    m_InDataStorageChanged = false;
  }
}

void QmitkFunctionalityWithoutStdMultiWidget::FireNodeSelected( mitk::DataNode* node )
{
  std::vector<mitk::DataNode*> nodes;
  nodes.push_back(node);
  this->FireNodesSelected(nodes);
}

void QmitkFunctionalityWithoutStdMultiWidget::FireNodesSelected( std::vector<mitk::DataNode*> nodes )
{
  if( !m_SelectionProvider )
    return;

  std::vector<mitk::DataNode::Pointer> nodesSmartPointers;
  for (std::vector<mitk::DataNode*>::iterator it = nodes.begin()
    ; it != nodes.end(); it++)
  {
    nodesSmartPointers.push_back( *it );
  }
  m_SelectionProvider->FireNodesSelected(nodesSmartPointers);

}



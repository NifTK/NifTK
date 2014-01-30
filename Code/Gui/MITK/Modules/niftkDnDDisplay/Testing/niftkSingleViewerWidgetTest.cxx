/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSingleViewerWidgetTest.h"

#include <QApplication>
#include <QSignalSpy>
#include <QTest>
#include <QTextStream>

#include <mitkGlobalInteraction.h>
#include <mitkIOUtil.h>
#include <mitkStandaloneDataStorage.h>
#include <mitkTestingMacros.h>

#include <QmitkRenderingManagerFactory.h>
#include <QmitkApplicationCursor.h>

#include <mitkNifTKCoreObjectFactory.h>
#include <niftkSingleViewerWidget.h>
#include <niftkMultiViewerWidget.h>
#include <niftkMultiViewerVisibilityManager.h>


class niftkSingleViewerWidgetTestClassPrivate
{
public:
  std::string FileName;
  mitk::DataStorage::Pointer DataStorage;
  mitk::RenderingManager::Pointer RenderingManager;

  mitk::DataNode::Pointer ImageNode;

  niftkSingleViewerWidget* Viewer;
  niftkMultiViewerVisibilityManager* VisibilityManager;

  bool InteractiveMode;
};


// --------------------------------------------------------------------------
niftkSingleViewerWidgetTestClass::niftkSingleViewerWidgetTestClass()
: QObject()
, d_ptr(new niftkSingleViewerWidgetTestClassPrivate())
{
  Q_D(niftkSingleViewerWidgetTestClass);

  d->ImageNode = 0;
  d->Viewer = 0;
  d->VisibilityManager = 0;
  d->InteractiveMode = false;
}


// --------------------------------------------------------------------------
niftkSingleViewerWidgetTestClass::~niftkSingleViewerWidgetTestClass()
{
}


// --------------------------------------------------------------------------
std::string niftkSingleViewerWidgetTestClass::GetFileName() const
{
  Q_D(const niftkSingleViewerWidgetTestClass);
  return d->FileName;
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::SetFileName(const std::string& fileName)
{
  Q_D(niftkSingleViewerWidgetTestClass);
  d->FileName = fileName;
}


// --------------------------------------------------------------------------
bool niftkSingleViewerWidgetTestClass::GetInteractiveMode() const
{
  Q_D(const niftkSingleViewerWidgetTestClass);
  return d->InteractiveMode;
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::SetInteractiveMode(bool interactiveMode)
{
  Q_D(niftkSingleViewerWidgetTestClass);
  d->InteractiveMode = interactiveMode;
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::initTestCase()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  // Need to load images, specifically using MIDAS/DRC object factory.
  ::RegisterNifTKCoreObjectFactory();

  mitk::GlobalInteraction* globalInteraction =  mitk::GlobalInteraction::GetInstance();
  globalInteraction->Initialize("global");
  globalInteraction->GetStateMachineFactory()->LoadBehaviorString(mitk::DnDDisplayStateMachine::STATE_MACHINE_XML);

  /// Create and register RenderingManagerFactory for this platform.
  static QmitkRenderingManagerFactory qmitkRenderingManagerFactory;
  Q_UNUSED(qmitkRenderingManagerFactory);

  /// Create one instance
  static QmitkApplicationCursor globalQmitkApplicationCursor;
  Q_UNUSED(globalQmitkApplicationCursor);

  d->DataStorage = mitk::StandaloneDataStorage::New();

  d->RenderingManager = mitk::RenderingManager::GetInstance();
  d->RenderingManager->SetDataStorage(d->DataStorage);

  std::vector<std::string> files;
  files.push_back(d->FileName);

  mitk::IOUtil::LoadFiles(files, *(d->DataStorage.GetPointer()));
  mitk::DataStorage::SetOfObjects::ConstPointer allImages = d->DataStorage->GetAll();
  MITK_TEST_CONDITION_REQUIRED(mitk::Equal(allImages->size(), 1), ".. Test image loaded.");

  d->ImageNode = (*allImages)[0];

  d->VisibilityManager = new niftkMultiViewerVisibilityManager(d->DataStorage);
  d->VisibilityManager->SetInterpolationType(DNDDISPLAY_CUBIC_INTERPOLATION);
  d->VisibilityManager->SetDefaultWindowLayout(WINDOW_LAYOUT_CORONAL);
  d->VisibilityManager->SetDropType(DNDDISPLAY_DROP_SINGLE);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::cleanupTestCase()
{
  Q_D(niftkSingleViewerWidgetTestClass);
  delete d->VisibilityManager;
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::init()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  d->Viewer = new niftkSingleViewerWidget(tr("QmitkRenderWindow"),
    -5, 20, 0, d->RenderingManager, d->DataStorage);
  d->Viewer->setObjectName(tr("niftkSingleViewerWidget"));

//  QColor backgroundColour("black");
  QColor backgroundColour("#fffaf0");
  d->Viewer->SetDirectionAnnotationsVisible(true);
  d->Viewer->SetBackgroundColor(backgroundColour);
  d->Viewer->SetShow3DWindowIn2x2WindowLayout(true);
  d->Viewer->SetRememberSettingsPerWindowLayout(true);
  d->Viewer->SetDisplayInteractionsEnabled(true);
  d->Viewer->SetCursorPositionsBound(true);
  d->Viewer->SetScaleFactorsBound(true);
  d->Viewer->SetDefaultSingleWindowLayout(WINDOW_LAYOUT_CORONAL);
  d->Viewer->SetDefaultMultiWindowLayout(WINDOW_LAYOUT_ORTHO);

  d->VisibilityManager->connect(d->Viewer, SIGNAL(NodesDropped(niftkSingleViewerWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)), SLOT(OnNodesDropped(niftkSingleViewerWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)), Qt::DirectConnection);

  d->VisibilityManager->RegisterViewer(d->Viewer);
  d->VisibilityManager->SetAllNodeVisibilityForViewer(0, false);

  d->Viewer->resize(1024, 768);
  d->Viewer->show();

  std::vector<mitk::DataNode*> nodes(1);
  nodes[0] = d->ImageNode;

  QmitkRenderWindow* axialWindow = d->Viewer->GetAxialWindow();
  this->dropNodes(axialWindow, nodes);

  d->Viewer->SetCursorVisible(true);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::cleanup()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  d->VisibilityManager->DeRegisterAllViewers();

  delete d->Viewer;
  d->Viewer = 0;
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::dropNodes(QWidget* window, const std::vector<mitk::DataNode*>& nodes)
{
  Q_D(niftkSingleViewerWidgetTestClass);

  QMimeData* mimeData = new QMimeData;
  QString dataNodeAddresses("");
  for (int i = 0; i < nodes.size(); ++i)
  {
    long dataNodeAddress = reinterpret_cast<long>(nodes[i]);
    QTextStream(&dataNodeAddresses) << dataNodeAddress;

    if (i != nodes.size() - 1)
    {
      QTextStream(&dataNodeAddresses) << ",";
    }
  }
  mimeData->setData("application/x-mitk-datanodes", QByteArray(dataNodeAddresses.toAscii()));
  QStringList types;
  types << "application/x-mitk-datanodes";
  QDropEvent dropEvent(window->rect().center(), Qt::CopyAction | Qt::MoveAction, mimeData, Qt::LeftButton, Qt::NoModifier);
  dropEvent.acceptProposedAction();
  QApplication::instance()->sendEvent(window, &dropEvent);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testViewer()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  QTest::qWaitForWindowShown(d->Viewer);

  if (d->InteractiveMode)
  {
    QEventLoop loop;
    loop.connect(d->Viewer, SIGNAL(destroyed()), SLOT(quit()));
    loop.exec();
  }

  QVERIFY(true);
}


// --------------------------------------------------------------------------
static void ShiftArgs(int& argc, char* argv[], int steps = 1)
{
  /// We exploit that there must be a NULL pointer after the arguments.
  /// (Guaranteed by the standard.)
  int i = 1;
  do
  {
    argv[i] = argv[i + steps];
    ++i;
  }
  while (argv[i - 1]);
  argc -= steps;
}


// --------------------------------------------------------------------------
int niftkSingleViewerWidgetTest(int argc, char* argv[])
{
  QApplication app(argc, argv);
  Q_UNUSED(app);

  niftkSingleViewerWidgetTestClass test;

  std::string interactiveModeOption("-i");
  for (int i = 1; i < argc; ++i)
  {
    if (std::string(argv[i]) == interactiveModeOption)
    {
      test.SetInteractiveMode(true);
      ::ShiftArgs(argc, argv);
      break;
    }
  }

  if (argc < 2)
  {
    MITK_INFO << "Missing argument. No image file given.";
    return 1;
  }

  test.SetFileName(argv[1]);

  /// We used the arguments to initialise the test. No arguments is passed
  /// to the Qt test, so that all the test functions are executed.
  argc = 1;
  argv[1] = NULL;
  return QTest::qExec(&test, argc, argv);
}

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMultiViewerWidgetTest.h"

#include <QApplication>
#include <QSignalSpy>
#include <QTest>
#include <QTextStream>

#include <mitkGlobalInteraction.h>
#include <mitkIOUtil.h>
#include <mitkStandaloneDataStorage.h>
#include <mitkTestingMacros.h>

#include <QmitkRegisterClasses.h>

#include <mitkNifTKCoreObjectFactory.h>
#include <niftkSingleViewerWidget.h>
#include <niftkMultiViewerWidget.h>
#include <niftkMultiViewerVisibilityManager.h>


class niftkMultiViewerWidgetTestClassPrivate
{
public:
  std::string FileName;
  mitk::DataStorage::Pointer DataStorage;
  mitk::RenderingManager::Pointer RenderingManager;

  mitk::DataNode::Pointer ImageNode;

  niftkMultiViewerWidget* MultiViewer;
  niftkMultiViewerVisibilityManager::Pointer VisibilityManager;

  bool InteractiveMode;
};


// --------------------------------------------------------------------------
niftkMultiViewerWidgetTestClass::niftkMultiViewerWidgetTestClass()
: QObject()
, d_ptr(new niftkMultiViewerWidgetTestClassPrivate())
{
}


// --------------------------------------------------------------------------
niftkMultiViewerWidgetTestClass::~niftkMultiViewerWidgetTestClass()
{
}


// --------------------------------------------------------------------------
std::string niftkMultiViewerWidgetTestClass::GetFileName() const
{
  Q_D(const niftkMultiViewerWidgetTestClass);
  return d->FileName;
}


// --------------------------------------------------------------------------
void niftkMultiViewerWidgetTestClass::SetFileName(const std::string& fileName)
{
  Q_D(niftkMultiViewerWidgetTestClass);
  d->FileName = fileName;
}


// --------------------------------------------------------------------------
bool niftkMultiViewerWidgetTestClass::GetInteractiveMode() const
{
  Q_D(const niftkMultiViewerWidgetTestClass);
  return d->InteractiveMode;
}


// --------------------------------------------------------------------------
void niftkMultiViewerWidgetTestClass::SetInteractiveMode(bool interactiveMode)
{
  Q_D(niftkMultiViewerWidgetTestClass);
  d->InteractiveMode = interactiveMode;
}


// --------------------------------------------------------------------------
void niftkMultiViewerWidgetTestClass::initTestCase()
{
  Q_D(niftkMultiViewerWidgetTestClass);

  // Need to load images, specifically using MIDAS/DRC object factory.
  ::RegisterNifTKCoreObjectFactory();

  QmitkRegisterClasses();

  d->DataStorage = mitk::StandaloneDataStorage::New();

  d->RenderingManager = mitk::RenderingManager::GetInstance();
  d->RenderingManager->SetDataStorage(d->DataStorage);

  std::vector<std::string> files;
  files.push_back(d->FileName);

  mitk::IOUtil::LoadFiles(files, *(d->DataStorage.GetPointer()));
  mitk::DataStorage::SetOfObjects::ConstPointer allImages = d->DataStorage->GetAll();
  MITK_TEST_CONDITION_REQUIRED(mitk::Equal(allImages->size(), 1), ".. Test image loaded.");

  d->ImageNode = (*allImages)[0];

  d->VisibilityManager = niftkMultiViewerVisibilityManager::New(d->DataStorage);
  d->VisibilityManager->SetInterpolationType(DNDDISPLAY_CUBIC_INTERPOLATION);
  d->VisibilityManager->SetDefaultWindowLayout(WINDOW_LAYOUT_CORONAL);
  d->VisibilityManager->SetDropType(DNDDISPLAY_DROP_SINGLE);
}


// --------------------------------------------------------------------------
void niftkMultiViewerWidgetTestClass::cleanupTestCase()
{
  Q_D(niftkMultiViewerWidgetTestClass);
  d->VisibilityManager = 0;
}


// --------------------------------------------------------------------------
void niftkMultiViewerWidgetTestClass::init()
{
  Q_D(niftkMultiViewerWidgetTestClass);

  // Create the niftkMultiViewerWidget
  d->MultiViewer = new niftkMultiViewerWidget(d->VisibilityManager, d->RenderingManager);

  // Setup GUI a bit more.
  d->MultiViewer->SetDropType(DNDDISPLAY_DROP_SINGLE);
  d->MultiViewer->SetShowOptionsVisible(true);
  d->MultiViewer->SetWindowLayoutControlsVisible(true);
  d->MultiViewer->SetViewerNumberControlsVisible(true);
  d->MultiViewer->SetShowDropTypeControls(false);
  d->MultiViewer->SetCursorDefaultVisibility(true);
  d->MultiViewer->SetDirectionAnnotationsVisible(true);
  d->MultiViewer->SetShow3DWindowIn2x2WindowLayout(false);
  d->MultiViewer->SetShowMagnificationSlider(true);
  d->MultiViewer->SetRememberSettingsPerWindowLayout(true);
  d->MultiViewer->SetSliceTracking(true);
  d->MultiViewer->SetTimeStepTracking(true);
  d->MultiViewer->SetMagnificationTracking(true);
  d->MultiViewer->SetDefaultWindowLayout(WINDOW_LAYOUT_CORONAL);

  d->MultiViewer->resize(1024, 768);
  d->MultiViewer->show();

  std::vector<mitk::DataNode*> nodes(1);
  nodes[0] = d->ImageNode;

  QmitkRenderWindow* axialWindow = d->MultiViewer->GetSelectedViewer()->GetAxialWindow();
  this->dropNodes(axialWindow, nodes);
}


// --------------------------------------------------------------------------
void niftkMultiViewerWidgetTestClass::cleanup()
{
  Q_D(niftkMultiViewerWidgetTestClass);
  delete d->MultiViewer;
  d->MultiViewer = 0;
}


// --------------------------------------------------------------------------
void niftkMultiViewerWidgetTestClass::dropNodes(QWidget* window, const std::vector<mitk::DataNode*>& nodes)
{
  Q_D(niftkMultiViewerWidgetTestClass);

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
void niftkMultiViewerWidgetTestClass::testViewer()
{
  Q_D(niftkMultiViewerWidgetTestClass);

  QTest::qWaitForWindowShown(d->MultiViewer);

  /// Remove the comment signs while you are doing interactive testing.
  if (d->InteractiveMode)
  {
    QEventLoop loop;
    loop.connect(d->MultiViewer, SIGNAL(destroyed()), SLOT(quit()));
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
int niftkMultiViewerWidgetTest(int argc, char* argv[])
{
  QApplication app(argc, argv);
  Q_UNUSED(app);

  niftkMultiViewerWidgetTestClass test;

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

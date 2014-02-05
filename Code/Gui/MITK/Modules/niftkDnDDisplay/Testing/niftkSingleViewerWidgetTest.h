/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkSingleViewerWidgetTest_h
#define __niftkSingleViewerWidgetTest_h

#include <QObject>

#include <mitkAtomicStateTransitionTester.cxx>

#include <niftkSingleViewerWidgetState.h>

#include <vector>

class niftkSingleViewerWidgetTestClassPrivate;

class niftkSingleViewerWidget;

namespace mitk
{
class DataNode;
}

class niftkSingleViewerWidgetTestClass: public QObject
{
  Q_OBJECT

public:

  typedef mitk::AtomicStateTransitionTester<const niftkSingleViewerWidget*, niftkSingleViewerWidgetState> ViewerStateTester;

  /// \brief Constructs a niftkSingleViewerWidgetTestClass object.
  explicit niftkSingleViewerWidgetTestClass();

  /// \brief Destructs the niftkSingleViewerWidgetTestClass object.
  virtual ~niftkSingleViewerWidgetTestClass();

  /// \brief Gets the name of the image file to load into the viewer.
  std::string GetFileName() const;

  /// \brief Sets the name of the image file to load into the viewer.
  void SetFileName(const std::string& fileName);

  /// \brief Gets the interactive mode.
  /// In interactive mode the windows are not closed when the test is finished.
  bool GetInteractiveMode() const;

  /// \brief Sets the interactive mode.
  /// In interactive mode the windows are not closed when the test is finished.
  void SetInteractiveMode(bool interactiveMode);

private slots:

  /// \brief Initialisation before the first test function.
  void initTestCase();

  /// \brief Clean up after the last test function.
  void cleanupTestCase();

  /// \brief Initialisation before each test function.
  void init();

  /// \brief Clean up after each test function.
  void cleanup();

  /// \brief Creates a viewer and and loads an image.
  void testViewer();

  /// \brief Tests if the selected orientation is correct after the image is loaded.
  void testGetOrientation();

  /// \brief Tests if the selected position is correct after the image is loaded.
  void testGetWindowLayout();

  /// \brief Tests if the correct render window is selected after the image is loaded.
  void testGetSelectedRenderWindow();

  /// \brief Tests if the correct renderer is focused after the image is loaded.
  void testFocusedRenderer();

  /// \brief Tests if the selected position is in the centre after the image is loaded.
  void testGetSelectedPosition();

  /// \brief Tests the SetSelectedPosition function.
  void testSetSelectedPosition();

  /// \brief Tests selecting a position by interaction (left mouse button click).
  void testSelectPositionByInteraction();

  /// \brief Tests the window layout change.
  void testSetWindowLayout();

private:

  void dropNodes(QWidget* window, const std::vector<mitk::DataNode*>& nodes);

  QScopedPointer<niftkSingleViewerWidgetTestClassPrivate> d_ptr;

  Q_DECLARE_PRIVATE(niftkSingleViewerWidgetTestClass)
  Q_DISABLE_COPY(niftkSingleViewerWidgetTestClass)
};


int niftkSingleViewerWidgetTest(int argc, char* argv[]);

#endif

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMultiViewerWidgetTest_h
#define niftkMultiViewerWidgetTest_h

#include <QObject>

#include <vector>

namespace mitk
{
class DataNode;
}

namespace niftk
{
class MultiViewerWidgetTestClassPrivate;


class MultiViewerWidgetTestClass: public QObject
{
  Q_OBJECT

public:

  explicit MultiViewerWidgetTestClass();
  virtual ~MultiViewerWidgetTestClass();

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

private:

  void dropNodes(QWidget* window, const std::vector<mitk::DataNode*>& nodes);

  QScopedPointer<MultiViewerWidgetTestClassPrivate> d_ptr;

  Q_DECLARE_PRIVATE(MultiViewerWidgetTestClass)
  Q_DISABLE_COPY(MultiViewerWidgetTestClass)
};

}

int niftkMultiViewerWidgetTest(int argc, char* argv[]);

#endif

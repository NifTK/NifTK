/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkMultiViewerWidgetTest_h
#define __niftkMultiViewerWidgetTest_h

#include <QObject>

#include <vector>

class niftkMultiViewerWidgetTestClassPrivate;

namespace mitk
{
class DataNode;
}

class niftkMultiViewerWidgetTestClass: public QObject
{
  Q_OBJECT

public:

  explicit niftkMultiViewerWidgetTestClass();
  virtual ~niftkMultiViewerWidgetTestClass();

  void setFileName(const std::string& fileName);

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

  QScopedPointer<niftkMultiViewerWidgetTestClassPrivate> d_ptr;

  Q_DECLARE_PRIVATE(niftkMultiViewerWidgetTestClass)
  Q_DISABLE_COPY(niftkMultiViewerWidgetTestClass)
};


int niftkMultiViewerWidgetTest(int argc, char* argv[]);

#endif

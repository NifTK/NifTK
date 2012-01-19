/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKMIDASMULTIVIEWVISIBILITYMANAGER_H_
#define QMITKMIDASMULTIVIEWVISIBILITYMANAGER_H_

#include "niftkQmitkExtExports.h"

#include <QWidget>
#include "mitkDataStorage.h"
#include "QmitkMIDASSingleViewWidget.h"
#include "itkImage.h"

class QmitkMIDASRenderWindow;
class QmitkMIDASSingleViewWidget;

/**
 * \class QmitkMultiViewVisibilityManager
 * \brief Maintains a list of QmitkMIDASSingleViewWidget and coordinates visibility
 * properties by listening to NodeAdded and NodeChanged events.
 */
class NIFTKQMITKEXT_EXPORT QmitkMIDASMultiViewVisibilityManager : public QObject
{
  Q_OBJECT

public:

  enum MIDASDropType
  {
    MIDAS_DROP_TYPE_SINGLE = 0,
    MIDAS_DROP_TYPE_MULTIPLE = 1,
    MIDAS_DROP_TYPE_ALL = 2,
  };

  enum MIDASDefaultOrientationType
  {
    MIDAS_ORIENTATION_AXIAL,
    MIDAS_ORIENTATION_SAGITTAL,
    MIDAS_ORIENTATION_CORONAL,
    MIDAS_ORIENTATION_AS_ACQUIRED
  };

  enum MIDASDefaultInterpolationType
  {
    MIDAS_INTERPOLATION_NONE,
    MIDAS_INTERPOLATION_LINEAR,
    MIDAS_INTERPOLATION_CUBIC
  };

  /// \brief This class must have a mitk::DataStorage so it is injected in the constructor.
  QmitkMIDASMultiViewVisibilityManager(mitk::DataStorage::Pointer dataStorage);
  virtual ~QmitkMIDASMultiViewVisibilityManager();

  /// \brief Each new QmitkMIDASSingleViewWidget should be registered with this class, so this class can manage visibility properties.
  void RegisterWidget(QmitkMIDASSingleViewWidget *widget);

  /// \brief Given a window, will return the corresponding list index, or -1 if not found.
  unsigned int GetIndexFromWindow(QmitkMIDASRenderWindow* window);

  /// \brief Called when a DataStorage Add Event was emmitted and sets m_InDataStorageChanged to true and calls NodeAdded afterwards.
  void NodeAddedProxy(const mitk::DataNode* node);

  /// \brief Called when a DataStorage Change Event was emmitted and sets m_InDataStorageChanged to true and calls NodeChanged afterwards.
  void NodeChangedProxy(const mitk::DataNode* node);

  /// \brief Set the drop type, which controls the behaviour when an image or images are dropped into a single widget.
  void SetDropType(MIDASDropType dropType) { m_DropType = dropType; }

  /// \brief Get the drop type, which controls the behaviour when an image or images are dropped into a single widget.
  MIDASDropType GetDropType() const { return m_DropType; }

  /// \brief Sets the default interpolation type.
  void SetDefaultInterpolationType(MIDASDefaultInterpolationType interpolation) { m_DefaultInterpolation = interpolation; }

  /// \brief Returns the default interpolation type.
  MIDASDefaultInterpolationType GetDefaultInterpolationType() const { return m_DefaultInterpolation; }

  /// \brief Sets the default orientation.
  void SetDefaultOrientationType(MIDASDefaultOrientationType orientation) { m_DefaultOrientation = orientation; }

  /// \brief Returns the default orientation type.
  MIDASDefaultOrientationType GetDefaultOrientationType() const { return m_DefaultOrientation; }

  /// \brief When we switch layouts, or drop in thumbnail mode, we clear all windows.
  void ClearAllWindows();

public slots:

  /// \brief When nodes are dropped, we set all the default properties, and renderer specific visibility flags.
  void OnNodesDropped(QmitkMIDASRenderWindow *window, std::vector<mitk::DataNode*> nodes);

signals:

protected:

  /// \brief Called when a DataStorage Add event was emmitted and may be reimplemented by deriving classes.
  virtual void NodeAdded(const mitk::DataNode* node);

  /// \brief Called when a DataStorage Change event was emmitted and may be reimplemented by deriving classes.
  virtual void NodeChanged(const mitk::DataNode* node);

  /// \brief For a given window, effectively sets the rendering window specific visibility to false.
  virtual void RemoveNodesFromWindow(int windowIndex);

  /// \brief For a given window, effectively sets the rendering window specific visibility to true.
  virtual void AddNodeToWindow(int windowIndex, mitk::DataNode* node);

protected slots:

private:

  // For a node, will set rendering window specific visibility to false for all registered windows.
  void SetInitialNodeProperties(mitk::DataNode* node);

  // For a node, will set the rendering window specific visibility property to match the global visibility property, so you can toggle the image in the data manager.
  void UpdateNodeProperties(mitk::DataNode* node);

  // Works out the correct orientation from the data, and from the preferences.
  QmitkMIDASSingleViewWidget::MIDASViewOrientation GetOrientation(std::vector<mitk::DataNode*> nodes);

  // ITK templated method (accessed via MITK access macros) to work out the orientation in the XY plane.
  template<typename TPixel, unsigned int VImageDimension>
  void GetAsAcquiredOrientation(
      itk::Image<TPixel, VImageDimension>* itkImage,
      QmitkMIDASSingleViewWidget::MIDASViewOrientation &outputOrientation
      );

  // Will retrieve the correct geometry from a list of nodes.
  // If nodeIndex < 0 (for single drop case).
  //   Search for first available image
  //   Failing that, first geometry.
  // If node index >=0, and < nodes.size()
  //   Picks out the geometry of the object for that index.
  // Else
  //   Picks out the first geometry.
  mitk::TimeSlicedGeometry::Pointer GetGeometry(std::vector<mitk::DataNode*> nodes, unsigned int nodeIndex);

  // This object MUST be connected to a datastorage, hence it is passed in via the constructor.
  mitk::DataStorage::Pointer m_DataStorage;

  // We maintain a list of data nodes present in each window.
  // When a node is removed from the window, the list is cleared.
  std::vector< std::vector<mitk::DataNode*> > m_ListOfDataNodes;

  // Additionally, we manage a list of widgets, where m_ListOfDataNodes.size() == m_ListOfWidgets.size() should always be true.
  std::vector< QmitkMIDASSingleViewWidget* > m_ListOfWidgets;

  // Simply keeps track of whether we are currently processing an update to avoid repeated/recursive calls.
  bool m_InDataStorageChanged;

  // Keeps track of the current mode, as it effects the response when images are dropped, as images are spread over single, multiple or all windows.
  MIDASDropType m_DropType;

  // Keeps track of the default orientation, as it affects the response when images are dropped, as the image should be oriented axial, coronal, sagittal, or as acquired (as per the X-Y plane).
  MIDASDefaultOrientationType m_DefaultOrientation;

  // Keeps track of the default interpolation, as it affects the response when images are dropped, as the dropped image should switch to that interpolation type, although as it is a node based property will affect all windows.
  MIDASDefaultInterpolationType m_DefaultInterpolation;

};

#endif

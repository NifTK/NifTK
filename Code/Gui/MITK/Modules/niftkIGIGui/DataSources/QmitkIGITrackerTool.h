/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkIGITrackerTool_h
#define QmitkIGITrackerTool_h

#include "niftkIGIGuiExports.h"
#include "QmitkIGINiftyLinkDataSource.h"
#include <mitkNavigationDataLandmarkTransformFilter.h>
#include <mitkDataNode.h>
#include <mitkPointSet.h>

/**
 * \class QmitkIGITrackerTool
 * \brief Base class for IGI Tracker Tools.
 */
class NIFTKIGIGUI_EXPORT QmitkIGITrackerTool : public QmitkIGINiftyLinkDataSource
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkIGITrackerTool, QmitkIGINiftyLinkDataSource);
  mitkNewMacro2Param(QmitkIGITrackerTool, mitk::DataStorage*, NiftyLinkSocketObject *);

  /**
   * \brief Defined in base class, so we check that the data is in fact a NiftyLinkMessageType containing tracking data.
   * \see mitk::IGIDataSource::CanHandleData()
   */
  virtual bool CanHandleData(mitk::IGIDataType* data) const;

  /**
   * \brief Defined in base class, this is the method where we do any update based on the available data.
   * \see mitk::IGIDataSource::Update()
   */
  virtual bool Update(mitk::IGIDataType* data);

  /**
   * \brief Instruct the tracker tool to enable a given tool
   * \param toolName the name of the tool, normally the name of a rom file for example 8700338.rom.
   * \param enable if true we want to enable a tool, if false, we want to disable it.
   */
  void EnableTool(const QString &toolName, const bool& enable);

  /**
   * \brief Instruct the tracker tool to retrieve the current position.
   * \param toolName the name of the tool, normally the name of a rom file for example 8700338.rom.
   */
  void GetToolPosition(const QString &toolName);

  /**
   * \brief Associates a dataNode with a given tool name, where many nodes can be associated with a single tool.
   * \return True if data node was successfully added, false if not.
   */
  bool AddDataNode(const QString toolName, mitk::DataNode::Pointer dataNode);
 
  /**
   * \brief Removes a dataNode from the associated list for the  given tool name, where many nodes can be associated with a single tool.
   * \return True if data node successfully removed
   */
  bool RemoveDataNode(const QString toolName, mitk::DataNode::Pointer dataNode);
  
  /**
   * \brief Return a QList of the tools associated with a given toolName
   */
  QList<mitk::DataNode::Pointer>  GetDataNode(const QString);

  /**
   * \brief Associates, with a pre-matrix, a dataNode with a given tool name, where many nodes can be associated with a single tool.
   * \return True if data node was successfully added, false if not.
   */
  bool AddPreMatrixDataNode(const QString toolName, mitk::DataNode::Pointer dataNode);
 
  /**
   * \brief Removes a dataNode from the associated (pre-matrix) list for the  given tool name, where many nodes can be associated with a single tool.
   * \return True if data node successfully removed
   */
  bool RemovePreMatrixDataNode(const QString toolName, mitk::DataNode::Pointer dataNode);
  
  /**
   * \brief Return a QList of the tools associated (pre-matrix) with a given toolName
   */
  QList<mitk::DataNode::Pointer>  GetPreMatrixDataNode(const QString);

  /**
   * \brief Not Widely Used: Set a flag to say we are doing ICP.
   */
  itkSetMacro(UseICP, bool);
  itkGetMacro(UseICP, bool);

  /**
   * \ brief get/set the VTK camera focal point
   */
  itkSetMacro(focalPoint, double);
  itkGetMacro(focalPoint, double);

  /** 
   * \ brief get/set whether to use fiducial transform filter
   */
  itkSetMacro(TransformTrackerToMITKCoords, bool);
  itkGetMacro(TransformTrackerToMITKCoords, bool);

  /**
   * \brief Not Widely Used: Erases the list of image and tracker fiducials, but leaves the nodes in data storage.
   */
  void ClearFiducials();

  /**
   * \brief Not Widely Used: Initialises the point sets and filters for point based registration, and adds them to data storage.
   */
  void InitializeFiducials();

  /**
   * \brief Not Widely Used: If not already added, adds the fiducials to data storage.
   */
  void AddFiducialsToDataStorage();

  /**
   * \brief Not Widely Used: If currently in data storage, will remove the fiducials from data storage.
   */
  void RemoveFiducialsFromDataStorage();

  /**
   * \brief Not Widely Used: Retrieves the internal point set representing image fiducials.
   */
  mitk::DataNode* GetImageFiducialsNode() const;

  /**
   * \brief Not Widely Used: Retrieves the internal point set representing tracker fiducials.
   */
  mitk::DataNode* GetTrackerFiducialsNode() const;

  /**
   * \brief Called by the add current tracker tip position button on QmitkFiducialRegistrationWidget to add the current position to tracker fiducial set
   */
  void GetCurrentTipPosition();

  /**
   * \brief Not Widely Used: Called from the "Register" button on the QmitkFiducialRegistrationWidget to register point sets.
   */
  void RegisterFiducials();

  /** 
   * \brief Applies the Fiducial transform to the passed data set
   */
  void ApplyFiducialTransform ( mitk::DataNode::Pointer );

  /**
   * \brief Not Widely Used: Retrieves / Creates tool, puts it into DataStorage, and returns pointer to the node.
   */
  mitk::DataNode* GetToolRepresentation(const QString name);

  /**
   * \brief Initialises the source based on the contents of the passed init string
   */
  void ProcessInitString(QString);

  /**
   * \brief Get the stored init string
   */
  QString GetInitString ();

  /**
   * \brief the state of the VTK camera link variable, 
   * true will move the vtk camera with the tracking tool
   */
  void SetCameraLink (bool);
  /**
   * \brief gets the state of the VTK camera link variable, 
   * true will move the vtk camera with the tracking tool
   */
  bool GetCameraLink ();

  /** 
   * \brief Sets up the fiducial landmark transform so that tracking transform
   * should be intuitive to use for fine alignment of model to lap lens
   */

  void SetUpPositioning (QString, mitk::DataNode::Pointer) ;

public slots:

  /**
   * \brief Main message handler routine for this tool, called by the signal from the socket.
   */
  virtual void InterpretMessage(NiftyLinkMessage::Pointer msg);

signals:

  void StatusUpdate(QString statusUpdateMessage);

protected:

  QmitkIGITrackerTool(mitk::DataStorage* storage, NiftyLinkSocketObject* socket); // Purposefully hidden.
  virtual ~QmitkIGITrackerTool(); // Purposefully hidden.

  QmitkIGITrackerTool(const QmitkIGITrackerTool&); // Purposefully not implemented.
  QmitkIGITrackerTool& operator=(const QmitkIGITrackerTool&); // Purposefully not implemented.

  /**
   * \brief \see IGIDataSource::SaveData();
   */
  virtual bool SaveData(mitk::IGIDataType* data, std::string& outputFileName);

private:

  /**
   * \brief Takes a message and extracts a matrix/transform and applies it.
   */
  void HandleTrackerData(NiftyLinkMessage* msg);

  /**
   * \brief Used to feedback a message of either coordinates, or matrices to
   * the GUI consolve via the StatusUpdate method, and also writes to console.
   */
  void DisplayTrackerData(NiftyLinkMessage* msg);

  /**
   * \brief Initialises m_PreMatrix with default values
   */
  void InitPreMatrix();
  

  QHash<QString, bool>                                 m_EnabledTools;
  QHash<QString, mitk::DataNode::Pointer>              m_ToolRepresentations;
  QHash<QString, mitk::DataNode::Pointer>              m_AssociatedTools;
  QHash<QString, mitk::DataNode::Pointer>              m_PreMatrixAssociatedTools;

  /** This lot is currently rarely used. */
  bool                                                 m_UseICP;
  bool                                                 m_PointSetsInitialized;
  bool                                                 m_LinkCamera;
  mitk::DataNode::Pointer                              m_ImageFiducialsDataNode;
  mitk::PointSet::Pointer                              m_ImageFiducialsPointSet;
  mitk::DataNode::Pointer                              m_TrackerFiducialsDataNode;
  mitk::PointSet::Pointer                              m_TrackerFiducialsPointSet;
  mitk::NavigationDataLandmarkTransformFilter::Pointer m_FiducialRegistrationFilter;   ///< this filter transforms from tracking coordinates into mitk world coordinates
  mitk::NavigationDataLandmarkTransformFilter::Pointer m_PermanentRegistrationFilter;  ///< this filter transforms from tracking coordinates into mitk world coordinates if needed it is interconnected before the FiducialEegistrationFilter
  
  //store a copy of the init string
  QString                                              m_InitString;
  double                                               m_focalPoint; //the focal point of the VTK camera used
  double                                               m_ClipNear; //the near clipping plane of the VTK camera used
  double                                               m_ClipFar; //the far clipping plane of the VTK camera used
  bool                                                 m_TransformTrackerToMITKCoords; //Set to true to use m_FiducialRegistrationFilter
  itk::Matrix<double,4,4>                              m_PreMatrix; //Use this to apply a matrix to a data node during tracking.
}; // end class

#endif


/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-07-25 07:31:59 +0100 (Wed, 25 Jul 2012) $
 Revision          : $Revision: 9401 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKIGITRACKERTOOL_H
#define QMITKIGITRACKERTOOL_H

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
  itkNewMacro(QmitkIGITrackerTool);
  mitkNewMacro1Param(QmitkIGITrackerTool,OIGTLSocketObject *);

  /**
   * \brief Defined in base class, so we check that the data is in fact a OIGTLMessageType containing tracking data.
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
   */
  void AddDataNode(const QString toolName, mitk::DataNode::Pointer dataNode);
  
  /**
   * \brief Return a QList of the tools associated with a given toolName
   */
  QList<mitk::DataNode::Pointer>  GetDataNode(const QString);

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
   * \brief Not Widely Used: Called from the "Register" button on the QmitkFiducialRegistrationWidget to register point sets.
   */
  void RegisterFiducials();

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
   * \sets the state of the VTK camera link variable, 
   * true will move the vtk camera with the tracking tool
   */
  void SetCameraLink (bool);
  /**
   * \gets the state of the VTK camera link variable, 
   * true will move the vtk camera with the tracking tool
   */
  bool GetCameraLink ();

public slots:

  /**
   * \brief Main message handler routine for this tool, called by the signal from the socket.
   */
  virtual void InterpretMessage(OIGTLMessage::Pointer msg);

signals:

  void StatusUpdate(QString statusUpdateMessage);

protected:

  QmitkIGITrackerTool(); // Purposefully hidden.
  virtual ~QmitkIGITrackerTool(); // Purposefully hidden.

  QmitkIGITrackerTool(const QmitkIGITrackerTool&); // Purposefully not implemented.
  QmitkIGITrackerTool(OIGTLSocketObject* socket); // Purposefully not implemented.
  QmitkIGITrackerTool& operator=(const QmitkIGITrackerTool&); // Purposefully not implemented.

  /**
   * \brief \see IGIDataSource::SaveData();
   */
  virtual bool SaveData(mitk::IGIDataType* data, std::string& outputFileName);

private:

  /**
   * \brief Takes a message and extracts a matrix/transform and applies it.
   */
  void HandleTrackerData(OIGTLMessage* msg);

  /**
   * \brief Used to feedback a message of either coordinates, or matrices to
   * the GUI consolve via the StatusUpdate method, and also writes to console.
   */
  void DisplayTrackerData(OIGTLMessage* msg);

  QHash<QString, bool>                                 m_EnabledTools;
  QHash<QString, mitk::DataNode::Pointer>              m_ToolRepresentations;
  QHash<QString, mitk::DataNode::Pointer>              m_AssociatedTools;

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
}; // end class

#endif


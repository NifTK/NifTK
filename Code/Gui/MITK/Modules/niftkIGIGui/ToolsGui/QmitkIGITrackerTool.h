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

  /**
   * \brief Defined in base class, so we check that the data is in fact a OIGTLMessageType;
   * \see mitk::IGIDataSource::CanHandleData()
   */
  virtual bool CanHandleData(mitk::IGIDataType::Pointer data) const;

  itkSetMacro(UseICP, bool);
  itkGetMacro(UseICP, bool);

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
   * \brief Add an associated tool to the named tool associated tools for a given 
   * tracker tool
   */
  void AddDataNode(const QString toolName, mitk::DataNode::Pointer dataNode);
  
  /**
   * \brief return a QList of the tools associated with a given toolName
   */

  QList<mitk::DataNode::Pointer>  GetDataNode(const QString);
  /**
   * \brief Erases the list of image and tracker fiducials, but leaves the nodes in data storage.
   */
  void ClearFiducials();

  /**
   * \Brief Initialises the point sets and filters for point based registration, and adds them to data storage.
   */
  void InitializeFiducials();

  /**
   * \brief If not already added, adds the fiducials to data storage.
   */
  void AddFiducialsToDataStorage();

  /**
   * \brief If currently in data storage, will remove the fiducials from data storage.
   */
  void RemoveFiducialsFromDataStorage();

  /**
   * \brief Retrieves the internal point set representing image fiducials.
   */
  mitk::DataNode* GetImageFiducialsNode() const;

  /**
   * \brief Retrieves the internal point set representing tracker fiducials.
   */
  mitk::DataNode* GetTrackerFiducialsNode() const;

  /**
   * \brief Called from the "Register" button on the QmitkFiducialRegistrationWidget to register point sets.
   */
  void RegisterFiducials();

  /**
   * \brief Retrieves / Creates tool, puts it into DataStorage, and returns pointer to the node.
   */
  mitk::DataNode* GetToolRepresentation(const QString name);
  
  /**
   * \brief If name does not end in ".rom", will add ".rom"
   */
  QString GetNameWithRom(const QString name);

public slots:

  /**
   * \brief Main message handler routine for this tool.
   */
  virtual void InterpretMessage(OIGTLMessage::Pointer msg);
  /**
   * \brief Finds a message which best matches id and handles it
   */
  virtual igtlUint64 HandleMessageByTimeStamp (igtlUint64 id);

signals:

  void StatusUpdate(QString statusUpdateMessage);

protected:

  QmitkIGITrackerTool(); // Purposefully hidden.
  virtual ~QmitkIGITrackerTool(); // Purposefully hidden.

  QmitkIGITrackerTool(const QmitkIGITrackerTool&); // Purposefully not implemented.
  QmitkIGITrackerTool& operator=(const QmitkIGITrackerTool&); // Purposefully not implemented.

private:

  void HandleTrackerData(OIGTLMessage::Pointer msg);

  /**
   * \brief Used to feedback a message of either coordinates, or matrices to
   * the GUI consolve via the StatusUpdate method, and also writes to console.
   */
  void DisplayTrackerData(OIGTLMessage::Pointer msg);

  unsigned long int                                    m_MsgCounter;
  QHash<QString, bool>                                 m_EnabledTools;
  QHash<QString, mitk::DataNode::Pointer>              m_ToolRepresentations;
  QHash<QString, mitk::DataNode::Pointer>              m_AssociatedTools;
  bool                                                 m_PointSetsInitialized;
  bool                                                 m_UseICP;
  mitk::DataNode::Pointer                              m_ImageFiducialsDataNode;
  mitk::PointSet::Pointer                              m_ImageFiducialsPointSet;
  mitk::DataNode::Pointer                              m_TrackerFiducialsDataNode;
  mitk::PointSet::Pointer                              m_TrackerFiducialsPointSet;
  mitk::NavigationDataLandmarkTransformFilter::Pointer m_FiducialRegistrationFilter;   ///< this filter transforms from tracking coordinates into mitk world coordinates
  mitk::NavigationDataLandmarkTransformFilter::Pointer m_PermanentRegistrationFilter;  ///< this filter transforms from tracking coordinates into mitk world coordinates if needed it is interconnected before the FiducialEegistrationFilter

}; // end class

#endif


/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef BreastSegmentationView_h
#define BreastSegmentationView_h

#include <berryISelectionListener.h>

#include <QmitkAbstractView.h>

#include "ui_BreastSegmentationViewControls.h"


/**
   \class BreastSegmentationView
   \brief GUI interface to perform a breast segmentation.

   \ingroup uk.ac.ucl.cmic.breastsegmentation
*/
class BreastSegmentationView : public QmitkAbstractView
{  
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT
  
  public:  

    static const std::string VIEW_ID;

    BreastSegmentationView();
    virtual ~BreastSegmentationView();

    void OnNodeAdded(const mitk::DataNode* node);
    void OnNodeRemoved(const mitk::DataNode* node);
    void OnNodeChanged(const mitk::DataNode* node);

  protected slots:
  
    // Callbacks from all the extra buttons not associated with mitk::Tool subclasses.
    void OnCancelButtonPressed();
    void OnExecuteButtonPressed();
    
  protected:

    /// \brief Called by framework, this method creates all the controls for this view
    virtual void CreateQtPartControl(QWidget *parent);

    /// \brief Called by framework, sets the focus on a specific widget.
    virtual void SetFocus();

    /// \brief Creation of the connections of widgets in this class and the slots in this class.
    virtual void CreateConnections();

    /// \brief Get the list of data nodes from the data manager
    mitk::DataStorage::SetOfObjects::ConstPointer GetNodes();

    /// \brief The specific controls for this widget
    Ui::BreastSegmentationViewControls m_Controls;

    /// \brief Flag indicating whether any factors influencing the segmentation have been modified
    bool m_Modified;

    /// \brief The number of Gaussian components or classes to segment
    int m_NumberOfGaussianComponents;

    /// \brief The number of multi-spectral MR components to segment
    int m_NumberOfMultiSpectralComponents;
    /// \brief The number of MR time-points to segment
    int m_NumberOfTimePoints;

    /// \brief The maximum number of iterations
    unsigned int m_MaximumNumberOfIterations;

    /// \brief The bias field correction order
    int m_BiasFieldCorrectionOrder;
    /// \brief
    float m_BiasFieldRatioThreshold;

    /** \brief The cost of having fat next to glandular tissue
	The costs of having the same label next to each other and of fat next to background is zero. */
    float m_AdiposeGlandularAdjacencyCost;
    /** \brief The cost of having background next to glandular tissue
	The costs of having the same label next to each other and of fat next to background is zero. */
    float m_BackgroundGlandularAdjacencyCost;
    

};

#endif // BreastSegmentationView_h


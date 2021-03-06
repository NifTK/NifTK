/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
#ifndef AffineTransformView_h
#define AffineTransformView_h

#include <QmitkAbstractView.h>
#include <niftkBaseView.h>
#include <QmitkRenderWindow.h>
#include <berryISelectionListener.h>

#include <itkImage.h>

#include <vtkInteractorStyle.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>
#include <vtkRenderWindowInteractor.h>

#include <mitkBaseGeometry.h>
#include <mitkBoundingObject.h>
#include <mitkDataNode.h>
#include <mitkDataStorage.h>
#include <mitkGlobalInteraction.h>
#include <mitkWeakPointer.h>

#include <niftkAffineTransformer.h>
#include <niftkAffineTransformParametersDataNodeProperty.h>
#include <niftkCustomVTKAxesActor.h>

#include "ui_AffineTransformViewControls.h"
#include "niftkAffineTransformDataInteractor3D.h"

/**
 * \class AffineTransformView
 * \brief Affine transform UI plugin, provides controls to rotate, translate,
 * scale and shear an mitk::DataNode's index to world geometry, which can be applied
 * to images, surfaces and meshes alike.  However, the Resample button only applies to images.
 *
 * \ingroup uk_ac_ucl_cmic_affinetransform_internal
 */
class AffineTransformView : public niftk::BaseView
{  
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT
  
  public:  

    AffineTransformView();
    virtual ~AffineTransformView();

  protected slots:

    /// \brief Called by framework, this method creates all the controls for this view
    virtual void CreateQtPartControl(QWidget *parent) override;

    /// \brief Called by framework, sets the focus on a specific widget.
    virtual void SetFocus() override;

    /** \brief Slot for all changes to transformation parameters. */
    void OnParameterChanged();

    /** \brief Slot for reset button that resets the parameter controls, and updates node geometry accordingly. */
    void OnResetTransformPushed();

    /** \brief Slot for saving transform to disk. */
    void OnSaveTransformPushed();

    /** \brief Slot for loading transform from disk. */
    void OnLoadTransformPushed();

    /** \brief Slot for resampling the current image. */
    void OnResampleTransformPushed();

    /** \brief Slot for updating the direction cosines of the current image */
    void OnApplyTransformPushed();

    /** \brief Slot for keeping the rotation sliders and spinboxes in synch*/
    void OnRotationValueChanged();

    /** \brief Slot for keeping the translation sliders and spinboxes in synch*/
    void OnTranslationValueChanged();

    /** \brief Slot for keeping the scaling sliders and spinboxes in synch*/
    void OnScalingValueChanged();

    /** \brief Slot for keeping the shearing sliders and spinboxes in synch*/
    void OnShearingValueChanged();

    //************************************************************************************************************************

    /** \brief Slot for switching between interactive and regular transformation editing */
    void OnInteractiveModeToggled(bool on);

     /** \brief Slot for switching between translation and rotation in interactive editing mode */
    void OnRotationToggled(bool on);

    /** \brief Slot to update display when the interactive alignment has finished */
    void OnTransformReady();

    /** \brief Slot for enabling / disabling fixed angle rotation / translations */
    void OnFixAngleToggled(bool on);

    /** \brief Slot for swithcing the main axis of translation / rotation */
    void OnAxisChanged(bool on);

    //************************************************************************************************************************

  protected:

    /// \see QmitkAbstractView::OnSelectionChanged.
    virtual void OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer> &nodes) override;

  private:

    /** Enables or Disables all the non-interactive controls. */
    void SetControlsEnabled(bool isEnabled);

    /** Enables or Disables all the interactive controls. */
    void SetInteractiveControlsEnabled(bool isEnabled);

    void SetSliderControlsEnabled(bool isEnabled);

    /** Sets the controls to the values given in the specific parameters property. */
    void SetUIValues(niftk::AffineTransformParametersDataNodeProperty::Pointer parametersProperty);

    /** Sets the controls to the Identity. */
    void ResetUIValues();

    /** Gets the values from the controls and stores them on the specified parametersProperty. */
    void GetValuesFromUI(niftk::AffineTransformParametersDataNodeProperty::Pointer parametersProperty);

    /** Gets the values from the displayed matrix and stores them in a vtkMatrix. */
    void GetValuesFromDisplay(vtkSmartPointer<vtkMatrix4x4> transform);

    /**
    * \brief Updates the displayed transform with the values from the spin-box controls.
    *
    * Uses the conventions of Ext/ITK/RegistrationToolbox/Transforms/itkEulerAffineTransform.txx / Ext/ITK/2D3DToolbox/Transforms/itkAffineTransform2D3D.txx:<br>
    * <ol>
    * <li>Change of CoR</li>
    * <li>Shears</li>
    * <li>Scaling</li>
    * <li>Rotations: \f$R = R_x\cdot R_y\cdot R_z\f$</li>
    * <li>Translation</li>
    * <li>Undo of CoR</li>
    * <ol>
    */
    void UpdateTransformDisplay();

     /** Resets the transformer. */
    void ResetAffineTransformer();

    //************************************************************************************************************************
    virtual void CreateNewBoundingObject(mitk::DataNode::Pointer);

    virtual void AddBoundingObjectToNode(mitk::DataNode::Pointer, bool fit);

    virtual void RemoveBoundingObjectFromNode();

    bool DisplayLegends(bool legendsON);
    //************************************************************************************************************************


private:
    Ui::AffineTransformWidget             * m_Controls;
    double                                  m_CentreOfRotation[3];
    mitk::DataNode::Pointer                 m_DataOwnerNode;
    niftk::AffineTransformer::Pointer        m_AffineTransformer;


    //************************************************************************************************************************
    bool                                    m_InInteractiveMode;
    bool                                    m_RotationMode;
    bool                                    m_LegendAdded;
    QWidget                               * m_ParentWidget;
    mitk::WeakPointer<mitk::BaseData>       m_CurrentDataObject;
    mitk::BoundingObject::Pointer           m_BoundingObject;
    mitk::DataNode::Pointer                 m_BoundingObjectNode;
    niftk::AffineTransformDataInteractor3D::Pointer  m_AffineDataInteractor3D;
    vtkLegendScaleActor                   * m_LegendActor;
    niftk::CustomVTKAxesActor             * m_CustomAxesActor;
    //************************************************************************************************************************
};

#endif // AffineTransformView_h


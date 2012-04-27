/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate$
 Revision          : $LastChangedRevision$
 Last modified by  : $LastModifiedByAuthor$

 Original author   : stian.johnsen.09@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
 
#ifndef AffineTransformView_h
#define AffineTransformView_h

#include "QmitkAbstractView.h"
#include "berryISelectionListener.h"
#include "vtkMatrix4x4.h"
#include "vtkSmartPointer.h"
#include "itkImage.h"

#include "ui_AffineTransformViewControls.h"
#include "mitkAffineTransformParametersDataNodeProperty.h"
#include "mitkDataNode.h"
#include "mitkDataStorage.h"

/**
 * \class AffineTransformView
 * \brief Affine transform UI plugin, provides controls to rotate, translate,
 * scale and shear an mitk::DataNode's index to world geometry, which can be applied
 * to images, surfaces and meshes alike.  However, the Resample button only applies to images.
 *
 * This class stores several AffineTransformDataNodeProperty on each data node:
 * <pre>
 * 1. The "Initial" transformation     = The transformation that was on the object, before this view added anything.
 *                                       So, if you load an image from file, this transformation is a copy of the geometry implied by the image header.
 *
 * 2. The "Incremental" transformation = Firstly, the dataNode->GetData()->GetGeometry() can only be composed with.
 *                                       So, we always have to calculate, for any change, the delta to be composed onto the existing transformation.
 *
 * 3. The "Pre-Loaded" transformation  = A transformation loaded from file.
 *                                       Loading a transformation from file also resets the GUI parameters.
 *                                       So, if you then add a rotation of 10 degrees about X axis, it is performed AFTER the transformation loaded from file.
 *
 * 4. The "Displayed" transformation   = The transformation that matches the GUI display.
 *                                       So, if you then add a rotation of 10 degrees about X axis, this transformation is just that.
 *
 * </pre>
 * and additionally a single AffineTransformParametersDataNodeProperty:
 * <pre>
 * 1. The "Displayed" parameters to match the "Displayed" transformation above.
 * </pre>
 * At no point are parameters derived or extracted from the affine transformation matrix,
 * as this is ambiguous and prone to numerical instability.
 *
 * \ingroup uk_ac_ucl_cmic_affinetransform_internal
 */
class AffineTransformView : public QmitkAbstractView
{  
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT
  
  public:  

    /// \brief Simply stores the view name = "uk.ac.ucl.cmic.affinetransformview"
    static const std::string VIEW_ID;

    /// \brief See class introduction.
    static const std::string INITIAL_TRANSFORM_KEY;

    /// \brief See class introduction.
    static const std::string INCREMENTAL_TRANSFORM_KEY;

    /// \brief See class introduction.
    static const std::string PRELOADED_TRANSFORM_KEY;

    /// \brief See class introduction.
    static const std::string DISPLAYED_TRANSFORM_KEY;

    /// \brief See class introduction.
    static const std::string DISPLAYED_PARAMETERS_KEY;

    AffineTransformView();
    virtual ~AffineTransformView();

  protected slots:

    /// \brief Called by framework, this method creates all the controls for this view
    virtual void CreateQtPartControl(QWidget *parent);

    /// \brief Called by framework, sets the focus on a specific widget.
    virtual void SetFocus();

    /** \brief Slot for all changes to transformation parameters. */
    void OnParameterChanged(const double);

    /** \brief Slot for radio button state changes. */
    void OnParameterChanged(const bool);

    /** \brief Slot for reset button that resets the parameter controls, and updates node geometry accordingly. */
    void OnResetTransformPushed();

    /** \brief Slot for saving transform to disk. */
    void OnSaveTransformPushed();

    /** \brief Slot for loading transform from disk. */
    void OnLoadTransformPushed();

    /** \brief Slot for resampling the current image. */
    void OnResampleTransformPushed();

  protected:

    /** \brief Computes a new linear transform (as 4x4 transform matrix) from the parameters set through the UI. */
    virtual vtkSmartPointer<vtkMatrix4x4> ComputeTransformFromParameters(void) const;

    /// \see QmitkAbstractView::OnSelectionChanged.
    virtual void OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer> &nodes);

  private:

    /** Enables or Disables all the controls. */
    void _SetControlsEnabled(bool isEnabled);

    /** Sets the controls to the values given in the specific parameters property. */
    void _SetControls(mitk::AffineTransformParametersDataNodeProperty &parametersProperty);

    /** Gets the values from the controls and stores them on the specified parametersProperty. */
    void _GetControls(mitk::AffineTransformParametersDataNodeProperty &parametersProperty);

    /** Sets the controls to the Identity, and doesn't update anything else. */
    void _ResetControls();

    /** Called by _InitialiseNodeProperties to initialise (to Identity) a specified transform property on a node. */
    void _InitialiseTransformProperty(std::string name, mitk::DataNode& node);

    /** Called by OnSelectionChanged to setup a node with default transformation properties, if it doesn't already have them. */
    void _InitialiseNodeProperties(mitk::DataNode& node);

    /** Called by _UpdateTransformationGeometry to set new transformations in the right properties of the node. */
    void _UpdateNodeProperties(
        const vtkSmartPointer<vtkMatrix4x4> displayedTransformFromParameters,
        const vtkSmartPointer<vtkMatrix4x4> incrementalTransformToBeComposed,
        mitk::DataNode& node);

    /** Called by _UpdateNodeProperties to update a transform property on a given node. */
    void _UpdateTransformProperty(std::string name, vtkSmartPointer<vtkMatrix4x4> transform, mitk::DataNode& node);

    /** The transform loaded from file is applied to the current node, and all its children, and it resets the GUI parameters to Identity, and hence the DISPLAY_TRANSFORM and DISPLAY_PARAMETERS to Identity.*/
    void _ApplyLoadedTransformToNode(const vtkSmartPointer<vtkMatrix4x4> transformFromFile, mitk::DataNode& node);

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
    void _UpdateTransformDisplay();

    /** \brief Updates the transform on the current node, and it's children. */
    void _UpdateTransformationGeometry();

    /** \brief Applies a re-sampling to the current node. */
    void _ApplyResampleToCurrentNode();

    Ui::AffineTransformWidget *m_Controls;
    double m_CentreOfRotation[3];
    mitk::DataNode::Pointer msp_DataOwnerNode;
};

#endif // AffineTransformView_h


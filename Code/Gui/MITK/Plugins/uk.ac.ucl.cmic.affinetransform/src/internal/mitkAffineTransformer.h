/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKAFFINETRANSFORMER_H
#define MITKAFFINETRANSFORMER_H

#include <itkObject.h>
#include <mitkBaseData.h>
#include <mitkDataStorage.h>
#include <mitkToolManager.h>
#include <mitkImage.h>
#include <vtkSmartPointer.h>
#include <vtkLinearTransform.h>
#include <mitkAffineTransformParametersDataNodeProperty.h>
#include <mitkAffineTransformDataNodeProperty.h>

namespace mitk {

/**
 * \brief Class to contain all the ITK/MITK logic for the Affine Transformation Plugin,
 * to separate from AffineTransformationView to make unit testing easier.
 *
 */
class AffineTransformer : public itk::Object
{

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

  //-----------------------------------------------------------------------------
  /// \brief MITK style class and New macros
  mitkClassMacro(AffineTransformer, itk::Object);
  itkNewMacro(AffineTransformer);

  itkGetMacro(RotateAroundCenter, bool);
  itkSetMacro(RotateAroundCenter, bool);

  /// \brief Sets the mitk::DataStorage on this object.
  void SetDataStorage(mitk::DataStorage::Pointer dataStorage);

  /// \brief Gets the DataStorage pointer from this object.
  mitk::DataStorage::Pointer GetDataStorage() const;

  /// \brief Get the transform parameters from the current data node
  mitk::AffineTransformParametersDataNodeProperty::Pointer
     GetCurrentTransformParameters() const;

  /// \brief Get a transform matrix from the current data node
  vtkSmartPointer<vtkMatrix4x4> GetTransformMatrixFromNode(std::string which) const;

  /// \brief Computes and returns the transformation matrix based on the current set of parameters
  vtkSmartPointer<vtkMatrix4x4> GetCurrentTransformMatrix() const;

  //-----------------------------------------------------------------------------
  /// \brief Called when a node changed.
  void OnNodeChanged(mitk::DataNode::Pointer node);

  /** \brief Slot for all changes to transformation parameters. */
  void OnParametersChanged(mitk::AffineTransformParametersDataNodeProperty::Pointer paramsProperty);

  /** \brief Slot for saving transform to disk. */
  void OnSaveTransform(std::string filename);

  /** \brief Slot for loading transform from disk. */
  void OnLoadTransform(std::string filename);

  /** \brief Slot for updating the direction cosines with the current transformation. */
  void OnApplyTransform(); //BIG TODO

  /** \brief Slot for resampling the current image. */
  void OnResampleTransform();

    /** Called by _InitialiseNodeProperties to initialise (to Identity) a specified transform property on a node. */
  void InitialiseTransformProperty(std::string name, mitk::DataNode::Pointer node);

  /** Called by OnSelectionChanged to setup a node with default transformation properties, if it doesn't already have them. */
  void InitialiseNodeProperties(mitk::DataNode::Pointer node);

  /** Called by _UpdateTransformationGeometry to set new transformations in the right properties of the node. */
  void UpdateNodeProperties(const vtkSmartPointer<vtkMatrix4x4> displayedTransformFromParameters,
                            const vtkSmartPointer<vtkMatrix4x4> incrementalTransformToBeComposed,
                            mitk::DataNode::Pointer);

  /** Called by _UpdateNodeProperties to update a transform property on a given node. */
  void UpdateTransformProperty(std::string name, vtkSmartPointer<vtkMatrix4x4> transform, mitk::DataNode::Pointer node);

  /** The transform loaded from file is applied to the current node, and all its children, and it resets the GUI parameters to Identity, and hence the DISPLAY_TRANSFORM and DISPLAY_PARAMETERS to Identity.*/
  void ApplyLoadedTransformToNode(const vtkSmartPointer<vtkMatrix4x4> transformFromFile, mitk::DataNode::Pointer node);

  /** \brief Applies a re-sampling to the current node. */
  void ApplyResampleToCurrentNode();

protected:

  AffineTransformer();
  virtual ~AffineTransformer();

  AffineTransformer(const AffineTransformer&); // Purposefully not implemented.
  AffineTransformer& operator=(const AffineTransformer&); // Purposefully not implemented.

  /// \brief Computes a new linear transform (as 4x4 transform matrix) from the parameters set through the UI.
  virtual vtkSmartPointer<vtkMatrix4x4> ComputeTransformFromParameters(void) const;

  /// \brief Updates the transform on the current node, and it's children.
  void UpdateTransformationGeometry();

private:
  /// \brief This member stores the actual transformation paramters updated from the UI
  mitk::AffineTransformParametersDataNodeProperty::Pointer m_CurrDispTransfProp;

  /// \brief This class needs a DataStorage to work.
  mitk::DataStorage::Pointer m_DataStorage;

  /// \brief Pointer to the current data node
  mitk::DataNode::Pointer    m_CurrentDataNode;

  // \brief Flag to set rotation around center
  bool                       m_RotateAroundCenter;

  /// \brief Stores the coordinates of the center of rotation
  double                     m_CentreOfRotation[3];

  /// \brief Stores the coordinates transformation parameters
  double                     m_Translation[3];
  double                     m_Rotation[3];
  double                     m_Scaling[3];
  double                     m_Shearing[3];
}; // end class

} // end namespace

#endif

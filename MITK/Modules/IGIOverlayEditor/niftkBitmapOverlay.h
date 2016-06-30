/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkBitmapOverlay_h
#define niftkBitmapOverlay_h

#include "niftkIGIOverlayEditorExports.h"
#include <mitkBaseData.h>
#include <mitkDataNode.h>
#include <mitkDataStorage.h>
#include <vtkSmartPointer.h>

class vtkRenderer;
class vtkRenderWindow;
class vtkMapper;
class vtkImageActor;
class vtkImageMapper;

namespace niftk
{

/**
 * \class BitmapOverlay
 * \brief Used to draw a 2D image into the background of a VTK Render Window.
 */
class NIFTKIGIOVERLAYEDITOR_EXPORT BitmapOverlay : public itk::Object
{
public:

  mitkClassMacroItkParent( BitmapOverlay, itk::Object );
  itkNewMacro( Self );

  /**
   * \brief Set a pointer to the mitk::DataStorage containing the image data for the overlay.
   */
  void SetDataStorage (mitk::DataStorage::Pointer);

  /**
   * \brief Returns the vtkRenderer responsible for rendering the image into the vtkRenderWindow.
   */
  virtual vtkRenderer* GetVtkRenderer();

  /**
   * \brief Sets the vtkRenderWindow in which the image will be shown.
   * Make sure, you have called this function before calling Enable()
   */
  virtual void SetRenderWindow( vtkRenderWindow* renderWindow );

  /**
   * \brief Returns the vtkRenderWindow, which is used for displaying the image.
   */
  itkGetMacro(RenderWindow, vtkRenderWindow*);

  /**
   * \brief Setter and Getter for Opacity.
   */
  void SetOpacity(const double& opacity);
  itkGetMacro(Opacity, double);

  /**
   * \brief Setter and Getter for whether to flip the view up vector of the VTK camera.
   */
  itkSetMacro(FlipViewUp, bool);
  itkGetMacro(FlipViewUp, bool);

  /**
   * \brief Checks if the image is currently enabled (visible)
   */
  virtual bool IsEnabled();

  /**
   * \brief Enables drawing of the image.
   * If you want to disable it, call the Disable() function.
   */
  virtual void Enable();

  /**
   * \brief Disables drawing of the image.
   * If you want to enable it, call the Enable() function.
   */
  virtual void Disable();

  /**
   * \brief if enable is true, calls Enable(), otherwise calls Disable().
   */
  virtual void SetEnabled(const bool& enable);

  /**
   * \brief Set the current data node to display as an overlay.
   * This method will check that the input is indeed an image.
   * \return true if successful and false otherwise.
   */
  virtual bool SetNode(const mitk::DataNode* node);

  /**
   * \brief For both foreground and background vtkRenderers, sets the
   * vtkCamera position so that the whole of the image is visible.
   */
  void SetupCamera();

  /**
   * \brief Called from niftk::Single3DViewWidget when a DataStorage Change Event was emitted.
   */
  void NodeChanged(const mitk::DataNode* node);

  /**
   * \brief Called from niftk::Single3DViewWidget when a DataStorage Node Added Event was emitted.
   */
  void NodeAdded(const mitk::DataNode* node);

  /**
   * \brief Called from niftk::Single3DViewWidget when a DataStorage Node Removed Event was emitted.
   */
  void NodeRemoved(const mitk::DataNode* node);

  void SetClippingRange(const double& nearZ, const double& farZ);
  void GetClippingRange(double& nearZ, double& farZ);

protected:

  BitmapOverlay(); // Purposefully hidden.
  virtual ~BitmapOverlay(); // Purposefully hidden.

  BitmapOverlay(const BitmapOverlay&); // Purposefully not implemented.
  BitmapOverlay& operator=(const BitmapOverlay&); // Purposefully not implemented.

private:

  // We don't own this, so neither do we delete this.
  vtkRenderWindow*               m_RenderWindow;

  vtkSmartPointer<vtkRenderer>   m_BackRenderer;
  vtkSmartPointer<vtkRenderer>   m_FrontRenderer;
  vtkSmartPointer<vtkImageActor> m_BackActor;
  vtkSmartPointer<vtkImageActor> m_FrontActor;

  mitk::DataStorage::Pointer     m_DataStorage;
  mitk::DataNode::Pointer        m_ImageDataNode;
  bool                           m_IsEnabled;
  double                         m_Opacity;
  bool                           m_FlipViewUp;
  double                         m_ClippingRange[2];
};

} // end namespace

#endif

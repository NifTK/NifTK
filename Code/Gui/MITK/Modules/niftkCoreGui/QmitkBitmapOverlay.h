/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkBitmapOverlay_h
#define QmitkBitmapOverlay_h

#include "niftkCoreGuiExports.h"
#include <mitkBaseData.h>
#include <mitkDataNode.h>
#include <mitkDataStorage.h>

class vtkRenderer;
class vtkRenderWindow;
class vtkMapper;
class vtkCamera;
class vtkImageActor;
class vtkImageMapper;

/**
 * \class QmitkBitmapOverlay
 * \brief Used to draw a 2D image into the background of a VTK Render Window.
 */
class NIFTKCOREGUI_EXPORT QmitkBitmapOverlay : public itk::Object
{
public:

  mitkClassMacro( QmitkBitmapOverlay, itk::Object );
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
   * \brief Setter and Getter for opacity.
   */
  itkSetMacro(Opacity, double);
  itkGetMacro(Opacity, double);

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


protected:

  QmitkBitmapOverlay(); // Purposefully hidden.
  virtual ~QmitkBitmapOverlay(); // Purposefully hidden.

  QmitkBitmapOverlay(const QmitkBitmapOverlay&); // Purposefully not implemented.
  QmitkBitmapOverlay& operator=(const QmitkBitmapOverlay&); // Purposefully not implemented.

private:

  /**
   * \brief Called when a DataStorage Change Event was emmitted.
   */
  void NodeChanged(const mitk::DataNode* node);

  /**
   * \brief Called when a DataStorage Node Added Event was emmitted.
   */
  void NodeAdded(const mitk::DataNode* node);

  /**
   * \brief Called when a DataStorage Node Removed Event was emmitted.
   */
  void NodeRemoved(const mitk::DataNode* node);

  /**
   * \brief Private method to ...
   */
  void SetupCamera();

  vtkRenderWindow*            m_RenderWindow;

  vtkRenderer*                m_BackRenderer;
  vtkRenderer*                m_FrontRenderer;
  vtkImageActor*              m_BackActor;
  vtkImageActor*              m_FrontActor;
  vtkImageMapper*             m_Mapper;
  vtkCamera*                  m_BackCamera;
  vtkCamera*                  m_FrontCamera;

  mitk::DataStorage::Pointer  m_DataStorage;
  bool                        m_IsEnabled;
  double                      m_Opacity;

  mitk::DataNode::Pointer     m_ImageDataNode;
  char *                      m_ImageData;
  bool                        m_UsingNVIDIA;
  mitk::Image::Pointer        m_ImageInNode;

};

#endif

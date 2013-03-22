/*===================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center,
Division of Medical and Biological Informatics.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.

See LICENSE.txt or http://www.mitk.org for details.

===================================================================*/

#ifndef _vtk_BitmapOverlay_h_
#define _vtk_BitmapOverlay_h_


#include <mitkBaseData.h>
#include <mitkDataNode.h>
#include <mitkDataStorage.h>
class vtkRenderer;
class vtkRenderWindow;
class vtkMapper;
class vtkCamera;
class vtkImageActor;
class vtkImageMapper;
class vtkLookupTable;
class vtkPolyData;
class vtkPNGReader;
class vtkImageImport;


class RenderWindow;
/**
 * Renders a company logo in the foreground
 * of a vtkRenderWindow.

 */
class BitmapOverlay : public mitk::BaseData
{
public:

  mitkClassMacro( BitmapOverlay, BaseData );

  itkNewMacro( Self );

  enum LogoPosition{ UpperLeft, UpperRight, LowerLeft, LowerRight, Middle };



  /**
   * Sets the renderwindow, in which the logo
   * will be shown. Make sure, you have called this function
   * before calling Enable()
   */
  virtual void SetRenderWindow( vtkRenderWindow* renderWindow );

  /**
   * Sets the source file for the logo.
   */
  virtual void SetLogoSource(const char* filename);
  /**
   * Sets the opacity level of the logo.
   */
  virtual void SetOpacity(double opacity);
  /**
   * Specifies the logo size, values from 0...10,
   * where 1 is a nice little logo
   */
  virtual void SetZoomFactor( double factor );

  /**
   * Enables drawing of the logo.
   * If you want to disable it, call the Disable() function.
   */
  virtual void Enable();

  /**
   * Disables drawing of the logo.
   * If you want to enable it, call the Enable() function.
   */
  virtual void Disable();

  /**
   * Checks, if the logo is currently
   * enabled (visible)
   */
  virtual bool IsEnabled();

  /**
   * Empty implementation, since the BitmapOverlay doesn't
   * support the requested region concept
   */
  virtual void SetRequestedRegionToLargestPossibleRegion();

  /**
   * Empty implementation, since the BitmapOverlay doesn't
   * support the requested region concept
   */
  virtual bool RequestedRegionIsOutsideOfTheBufferedRegion();

  /**
   * Empty implementation, since the BitmapOverlay doesn't
   * support the requested region concept
   */
  virtual bool VerifyRequestedRegion();

  /**
   * Empty implementation, since the BitmapOverlay doesn't
   * support the requested region concept
   */
  virtual void SetRequestedRegion(itk::DataObject*);

  /**
   * Returns the vtkRenderWindow, which is used
   * for displaying the logo
   */
  virtual vtkRenderWindow* GetRenderWindow();

  /**
   * Returns the renderer responsible for
   * rendering the logo into the
   * vtkRenderWindow
   */
  virtual vtkRenderer* GetVtkRenderer();

  /**
   * Returns the actor associated with the logo
   */
  virtual vtkImageActor* GetActor();

  /**
   * Returns the mapper associated with the logo
   */
  virtual vtkImageMapper* GetMapper();

  /**
   * If set true, this method forces the logo rendering mechanism that it always
   * renders the MBI department logo, independent from mainapp option settings.
   */
  virtual void ForceMBILogoVisible(bool visible);

  /**
   * Set a pointer to a data node containing image data for the overlay
   */
  void SetDataNode (mitk::DataNode::Pointer);
  /**
   * Set a pointer to a data storage  for the overlay
   */
  void SetDataStorage (mitk::DataStorage::Pointer);

 /// \brief Called when a DataStorage Change Event was emmitted and sets m_InDataStorageChanged to true and calls NodeChanged afterwards.
   void NodeChangedProxy(const mitk::DataNode* node);
protected:
  void SetupCamera();
  void SetupPosition();

  /**
   * Constructor
   */
  BitmapOverlay();

  /**
   * Destructor
   */
  ~BitmapOverlay();

  vtkRenderWindow*            m_RenderWindow;
  vtkRenderer*                m_BackRenderer;
  vtkRenderer*                m_FrontRenderer;
  vtkImageActor*              m_BackActor;
  vtkImageActor*              m_FrontActor;
  vtkImageMapper*             m_Mapper;
  vtkPNGReader*               m_PngReader;
  vtkCamera*                  m_BackCamera;
  vtkCamera*                  m_FrontCamera;
  vtkImageImport*             m_VtkImageImport;

  std::string                 m_FileName;

  bool                        m_IsEnabled;
  bool                        m_ForceShowMBIDepartmentLogo;

  LogoPosition                m_LogoPosition;
  double                      m_ZoomFactor;
  double                      m_Opacity;

  char *                      m_ImageData;

  mitk::DataNode::Pointer     m_ImageDataNode;
  mitk::DataStorage::Pointer  m_DataStorage;

};

#endif

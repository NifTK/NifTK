/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <mitkTestingMacros.h>
#include <mitkVector.h>
#include <mitkIOUtil.h>
#include <mitkDataStorage.h>
#include <mitkStandaloneDataStorage.h>
#include <mitkDataNode.h>
#include <mitkImageAccessByItk.h>
#include <mitkToolManager.h>
#include <mitkGeometry3D.h>
#include <mitkPositionEvent.h>
#include <mitkStateEvent.h>
#include <mitkBaseRenderer.h>
#include <mitkRenderingManager.h>
#include <mitkRenderWindow.h>
#include <mitkFocusManager.h>
#include <mitkGlobalInteraction.h>

#include "mitkNifTKCoreObjectFactory.h"
#include "mitkMIDASTool.h"
#include "mitkMIDASPaintbrushTool.h"
#include "mitkMIDASImageUtils.h"

/**
 * \brief Test class for mitkMIDASPaintbrushTool.
 */
class mitkMIDASPaintbrushToolClass
{

public:

  mitk::DataStorage::Pointer m_DataStorage;
  mitk::ToolManager::Pointer m_ToolManager;
  mitk::RenderWindow::Pointer m_RenderWindow;
  mitk::RenderingManager::Pointer m_RenderingManager;
  mitk::MIDASPaintbrushTool* m_Tool;
  int m_PaintbrushToolId;


  //-----------------------------------------------------------------------------
  void Setup(char* argv[])
  {
    std::string fileName = argv[1];

    // Need to load images, specifically using MIDAS/DRC object factory.
    RegisterNifTKCoreObjectFactory();
    mitk::GlobalInteraction::GetInstance()->Initialize("mitkMIDASPaintbrushToolClass");

    m_DataStorage = mitk::StandaloneDataStorage::New();
    m_ToolManager = mitk::ToolManager::New(m_DataStorage);

    // We load the same file 2 times, then rename volumes.
    std::vector<std::string> files;
    files.push_back(fileName);
    files.push_back(fileName);

    mitk::IOUtil::LoadFiles(files, *(m_DataStorage.GetPointer()));
    mitk::DataStorage::SetOfObjects::ConstPointer allImages = m_DataStorage->GetAll();
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(allImages->size(), 2),".. Testing 2 images loaded.");

    const mitk::DataNode::Pointer subtractionsNode = (*allImages)[0];
    subtractionsNode->SetName(mitk::MIDASTool::MORPH_EDITS_SUBTRACTIONS);

    const mitk::DataNode::Pointer additionsNode = (*allImages)[1];
    additionsNode->SetName(mitk::MIDASTool::MORPH_EDITS_ADDITIONS);

    mitk::ToolManager::DataVectorType vector;
    vector.push_back(additionsNode);
    vector.push_back(subtractionsNode);

    m_ToolManager->SetWorkingData(vector);
    m_Tool = dynamic_cast<mitk::MIDASPaintbrushTool*>(m_ToolManager->GetToolById(m_ToolManager->GetToolIdByToolType<mitk::MIDASPaintbrushTool>()));

    m_RenderingManager = mitk::RenderingManager::GetInstance();
    m_RenderingManager->SetDataStorage(m_DataStorage);

    m_RenderWindow = mitk::RenderWindow::New(NULL, "mitkMIDASPaintbrushToolClass", m_RenderingManager);

    m_RenderingManager->InitializeViews(m_DataStorage->ComputeVisibleBoundingGeometry3D());
  }

  //-----------------------------------------------------------------------------
  void TestToolPresent()
  {
    MITK_TEST_OUTPUT(<< "Starting TestToolPresent...");
    MITK_TEST_CONDITION( m_Tool != NULL, ".. Testing tool present");
    MITK_TEST_OUTPUT(<< "Finished TestToolPresent...");
  }

  //-------------------------------------------------------------------------------------
  // Spec: If the user clicks and releases - nothing is drawn.
  //       To get something drawn you have to move the mouse while holding down a button.
  //-------------------------------------------------------------------------------------
  void TestSingleClick(unsigned int imageId, unsigned int cursorSize, unsigned int expectedResult)
  {
    MITK_TEST_OUTPUT(<< "Starting TestSingleClick... image=" << imageId << ", cursorSize=" << cursorSize << ", expectedResult=" << expectedResult);

    // First blank image.
    mitk::ToolManager::DataVectorType workingData = m_ToolManager->GetWorkingData();
    mitk::DataNode::Pointer node = workingData[imageId];

    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(node->GetData());
    const mitk::Image* constImage = const_cast<const mitk::Image*>(image.GetPointer());

    MITK_TEST_CONDITION( image.IsNotNull(), ".. Testing image present");

    mitk::FillImage(image, 0);
    unsigned long int numberOfVoxelsInImage = mitk::GetNumberOfVoxels(constImage);
    unsigned long int voxelCounter = mitk::CountBetweenThreshold(constImage, -0.01, 0.01);
    MITK_TEST_CONDITION( voxelCounter == numberOfVoxelsInImage, ".. Testing image blank");

    // Get Middle Voxel, convert to millimetres, make position event.
    mitk::Point3D voxelIndex = mitk::GetMiddlePointInVoxels(image);
    mitk::PositionEvent event = mitk::GeneratePositionEvent(m_RenderWindow->GetRenderer(), constImage, voxelIndex);

    // Generate Left or Right mouse click event.
    if (imageId == 0)
    {
      int eventId = mitk::EIDLEFTMOUSEBTN;
      const mitk::StateEvent stateEvent(eventId, &event);
      m_Tool->OnLeftMousePressed(NULL, &stateEvent);
      m_Tool->OnLeftMouseReleased(NULL, &stateEvent);
    }
    else
    {
      int eventId = mitk::EIDRIGHTMOUSEBTN;
      const mitk::StateEvent stateEvent(eventId, &event);
      m_Tool->OnRightMousePressed(NULL, &stateEvent);
      m_Tool->OnRightMouseReleased(NULL, &stateEvent);
    }

    // Count voxels that got painted.
    voxelCounter = mitk::CountBetweenThreshold(constImage, 0.99, 1.01);

    // Compare with the expected number of voxels that got painted.
    MITK_TEST_OUTPUT(<<"Resulting voxel size=" << voxelCounter);
    MITK_TEST_CONDITION( voxelCounter == expectedResult, ".. Testing cross size");

    MITK_TEST_OUTPUT(<< "Finished TestSingleClick... image=" << imageId);
  }



  //-------------------------------------------------------------------------------------
  // Spec: If the user clicks and drags, and move by at least 1 voxel, we draw crosses.
  //       Background is zero, and the drawn crosses are voxel intensity 1.
  //-------------------------------------------------------------------------------------
  void TestClickDrag(unsigned int imageId, unsigned int cursorSize, unsigned int numberOfVoxelsDifference, unsigned int expectedResult)
  {
    MITK_TEST_OUTPUT(<< "Starting TestClickDrag... image=" << imageId << ", cursorSize=" << cursorSize << ", expectedResult=" << expectedResult);

    // First blank image.
    mitk::ToolManager::DataVectorType workingData = m_ToolManager->GetWorkingData();
    mitk::DataNode::Pointer node = workingData[imageId];
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(node->GetData());
    const mitk::Image* constImage = const_cast<const mitk::Image*>(image.GetPointer());
    mitk::FillImage(image, 0);

    // Set tool size
    m_Tool->SetCursorSize(cursorSize);

    unsigned long int numberOfVoxelsInImage = mitk::GetNumberOfVoxels(constImage);
    unsigned long int voxelCounter = mitk::CountBetweenThreshold(constImage, -0.01, 0.01);
    MITK_TEST_CONDITION( voxelCounter == numberOfVoxelsInImage, ".. Testing image blank");

    // Get Middle Voxel, convert to millimetres, make position event.
    mitk::Point3D voxelIndex = mitk::GetMiddlePointInVoxels(constImage);
    mitk::PositionEvent middlePositionEvent = mitk::GeneratePositionEvent(m_RenderWindow->GetRenderer(), constImage, voxelIndex);

    // Paint in XY plane, exactly diagonal.
    mitk::Point3D nextVoxelIndex = voxelIndex;
    nextVoxelIndex[0] += numberOfVoxelsDifference;
    nextVoxelIndex[1] += numberOfVoxelsDifference;

    // Create another position event.
    mitk::PositionEvent nextPositionEvent = mitk::GeneratePositionEvent(m_RenderWindow->GetRenderer(), constImage, nextVoxelIndex);

    // Generate Left or Right mouse click events.
    if (imageId == 0)
    {
      int eventId = mitk::EIDLEFTMOUSEBTN;
      const mitk::StateEvent stateEvent1(eventId, &middlePositionEvent);
      m_Tool->OnLeftMousePressed(NULL, &stateEvent1);

      const mitk::StateEvent stateEvent2(eventId, &nextPositionEvent);
      m_Tool->OnLeftMouseMoved(NULL, &stateEvent2);

      const mitk::StateEvent stateEvent3(eventId, &nextPositionEvent);
      m_Tool->OnLeftMouseReleased(NULL, &stateEvent3);
    }
    else
    {
      int eventId = mitk::EIDMIDDLEMOUSEBTN;
      const mitk::StateEvent stateEvent1(eventId, &middlePositionEvent);
      m_Tool->OnMiddleMousePressed(NULL, &stateEvent1);

      const mitk::StateEvent stateEvent2(eventId, &nextPositionEvent);
      m_Tool->OnMiddleMouseMoved(NULL, &stateEvent2);

      const mitk::StateEvent stateEvent3(eventId, &nextPositionEvent);
      m_Tool->OnMiddleMouseReleased(NULL, &stateEvent3);
    }

    // Count voxels that got painted.
    voxelCounter = mitk::CountBetweenThreshold(constImage, 0.99, 1.01);
    MITK_TEST_OUTPUT(<<"Resulting voxelCounter=" << voxelCounter << ", with expectedResult=" << expectedResult);
    MITK_TEST_CONDITION( voxelCounter == expectedResult, ".. Testing cross size");

    MITK_TEST_OUTPUT(<< "Finished TestClickDrag...");
  }


  //-----------------------------------------------------------------------------
  void TestErase()
  {
    MITK_TEST_OUTPUT(<< "Starting TestErase...");

    // First fill image 0 with 1.
    mitk::ToolManager::DataVectorType workingData = m_ToolManager->GetWorkingData();
    mitk::DataNode::Pointer node = workingData[1];
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(node->GetData());
    const mitk::Image* constImage = const_cast<const mitk::Image*>(image.GetPointer());
    mitk::FillImage(image, 1);

    // Set tool size
    m_Tool->SetCursorSize(2);

    unsigned long int numberOfVoxelsInImage = mitk::GetNumberOfVoxels(constImage);
    unsigned long int voxelCounter = mitk::CountBetweenThreshold(constImage, 0.99, 1.01);
    MITK_TEST_CONDITION( voxelCounter == numberOfVoxelsInImage, ".. Testing image all filled in");

    // Get Middle Voxel, convert to millimetres, make position event.
    mitk::Point3D voxelIndex = mitk::GetMiddlePointInVoxels(constImage);
    mitk::PositionEvent middlePositionEvent = mitk::GeneratePositionEvent(m_RenderWindow->GetRenderer(), constImage, voxelIndex);

    // Paint in XY plane, exactly diagonal.
    mitk::Point3D nextVoxelIndex = voxelIndex;
    nextVoxelIndex[0] += 1;
    nextVoxelIndex[1] += 1;

    // Create another position event.
    mitk::PositionEvent nextPositionEvent = mitk::GeneratePositionEvent(m_RenderWindow->GetRenderer(), constImage, nextVoxelIndex);

    // Generate Right mouse click events.
    int eventId = mitk::EIDRIGHTMOUSEBTN;
    const mitk::StateEvent stateEvent1(eventId, &middlePositionEvent);
    m_Tool->OnRightMousePressed(NULL, &stateEvent1);

    const mitk::StateEvent stateEvent2(eventId, &nextPositionEvent);
    m_Tool->OnRightMouseMoved(NULL, &stateEvent2);

    const mitk::StateEvent stateEvent3(eventId, &nextPositionEvent);
    m_Tool->OnRightMouseReleased(NULL, &stateEvent3);

    // Count voxels that got painted.
    voxelCounter = mitk::CountBetweenThreshold(constImage, 0.99, 1.01);
    unsigned int expectedResult = numberOfVoxelsInImage - 8;
    MITK_TEST_OUTPUT(<<"Resulting voxelCounter=" << voxelCounter << ", with expectedResult=" << expectedResult);
    MITK_TEST_CONDITION( voxelCounter == expectedResult, ".. Testing cross size");

    MITK_TEST_OUTPUT(<< "Finished TestErase...");
  }


  //-----------------------------------------------------------------------------
  void TestScan()
  {
    MITK_TEST_OUTPUT(<< "Starting TestScan...");

    // First blank image.
    mitk::ToolManager::DataVectorType workingData = m_ToolManager->GetWorkingData();
    mitk::DataNode::Pointer node = workingData[0];
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(node->GetData());
    const mitk::Image* constImage = const_cast<const mitk::Image*>(image.GetPointer());

    mitk::FillImage(image, 0);

    // Set tool size
    m_Tool->SetCursorSize(3);

    unsigned int sizeX = image->GetDimension(0);
    unsigned int sizeY = image->GetDimension(1);
    unsigned int sizeZ = image->GetDimension(2);

    for (unsigned int z = 0; z < sizeZ; z++)
    {
      for (unsigned int y = 0; y < sizeY; y++)
      {
        for (unsigned int x = 0; x < sizeX; x++)
        {

          mitk::Point3D voxelIndex;
          voxelIndex[0] = x;
          voxelIndex[1] = y;
          voxelIndex[2] = z;

          mitk::PositionEvent positionEvent = mitk::GeneratePositionEvent(m_RenderWindow->GetRenderer(), constImage, voxelIndex);

          // Paint in XY plane, exactly diagonal.
          mitk::Point3D nextVoxelIndex = voxelIndex;
          nextVoxelIndex[0] += 1;
          nextVoxelIndex[1] += 1;

          // Create another position event.
          mitk::PositionEvent nextPositionEvent = mitk::GeneratePositionEvent(m_RenderWindow->GetRenderer(), constImage, nextVoxelIndex);

          int eventId = mitk::EIDLEFTMOUSEBTN;
          const mitk::StateEvent stateEvent1(eventId, &positionEvent);
          m_Tool->OnLeftMousePressed(NULL, &stateEvent1);

          const mitk::StateEvent stateEvent2(eventId, &nextPositionEvent);
          m_Tool->OnLeftMouseMoved(NULL, &stateEvent2);

          const mitk::StateEvent stateEvent3(eventId, &nextPositionEvent);
          m_Tool->OnLeftMouseReleased(NULL, &stateEvent3);
        }
      }
    }

    // Count voxels that got painted.
    unsigned long int voxelCounter = mitk::CountBetweenThreshold(constImage, 0.99, 1.01);
    unsigned long int numberOfVoxelsInImage = mitk::GetNumberOfVoxels(constImage);
    MITK_TEST_OUTPUT(<<"Resulting voxelCounter=" << voxelCounter << ", with expectedResult=" << numberOfVoxelsInImage);
    MITK_TEST_CONDITION( voxelCounter == numberOfVoxelsInImage, ".. Testing painted all voxels");

    MITK_TEST_OUTPUT(<< "Finished TestScan...");
  }


  //-----------------------------------------------------------------------------
  void TestClickOutOfBounds(unsigned int x, unsigned int y, unsigned int z)
  {
    MITK_TEST_OUTPUT(<< "Starting TestClickOutOfBounds...x=" << x << ", " << y << ", " << z);

    mitk::ToolManager::DataVectorType workingData = m_ToolManager->GetWorkingData();
    mitk::DataNode::Pointer node = workingData[0];
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(node->GetData());
    const mitk::Image* constImage = const_cast<const mitk::Image*>(image.GetPointer());

    mitk::FillImage(image, 0);

    // Set tool size
    m_Tool->SetCursorSize(3);

    mitk::Point3D voxelIndex;
    voxelIndex[0] = x;
    voxelIndex[1] = y;
    voxelIndex[2] = z;

    mitk::PositionEvent positionEvent = mitk::GeneratePositionEvent(m_RenderWindow->GetRenderer(), constImage, voxelIndex);

    // Paint in XY plane, exactly diagonal.
    mitk::Point3D nextVoxelIndex = voxelIndex;
    nextVoxelIndex[0] += 1;
    nextVoxelIndex[1] += 1;

    // Create another position event.
    mitk::PositionEvent nextPositionEvent = mitk::GeneratePositionEvent(m_RenderWindow->GetRenderer(), constImage, nextVoxelIndex);

    int eventId = mitk::EIDLEFTMOUSEBTN;
    const mitk::StateEvent stateEvent1(eventId, &positionEvent);
    m_Tool->OnLeftMousePressed(NULL, &stateEvent1);

    const mitk::StateEvent stateEvent2(eventId, &nextPositionEvent);
    m_Tool->OnLeftMouseMoved(NULL, &stateEvent2);

    const mitk::StateEvent stateEvent3(eventId, &nextPositionEvent);
    m_Tool->OnLeftMouseReleased(NULL, &stateEvent3);

    // Count voxels that got painted, should be zero.
    unsigned long int voxelCounter = mitk::CountBetweenThreshold(constImage, 0.99, 1.01);
    MITK_TEST_OUTPUT(<<"Resulting voxelCounter=" << voxelCounter << ", with expectedResult=0");
    MITK_TEST_CONDITION( voxelCounter == 0, ".. Testing painted zero voxels");

    MITK_TEST_OUTPUT(<< "Finished TestClickOutOfBounds...");
  }


  //-----------------------------------------------------------------------------
  void TestClickJustOutOfBounds()
  {
    mitk::ToolManager::DataVectorType workingData = m_ToolManager->GetWorkingData();
    mitk::DataNode::Pointer node = workingData[0];
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(node->GetData());

    unsigned int sizeX = image->GetDimension(0);
    unsigned int sizeY = image->GetDimension(1);
    unsigned int sizeZ = image->GetDimension(2);

    this->TestClickOutOfBounds(sizeX, sizeY, sizeZ);
  }

  //-----------------------------------------------------------------------------
  void TestClickWayOutOfBounds()
  {
    unsigned int x = std::numeric_limits<unsigned int>::max();
    unsigned int y = std::numeric_limits<unsigned int>::max();
    unsigned int z = std::numeric_limits<unsigned int>::max();
    this->TestClickOutOfBounds(x, y, z);
  }

};

/**
 * Basic test harness for mitkMIDASPaintbrushToolScanTest.
 */
int mitkMIDASPaintbrushToolTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkMIDASPaintbrushToolTest");

  // We are testing specifically with image ${NIFTK_DATA_DIR}/Input/nv-11x11x11.nii which is 11x11x11.

  mitkMIDASPaintbrushToolClass *testClass = new mitkMIDASPaintbrushToolClass();
  testClass->Setup(argv);
  testClass->TestToolPresent();
  testClass->TestSingleClick(0, 2, 0);
  testClass->TestSingleClick(0, 4, 0);
  testClass->TestSingleClick(0, 6, 0);
  testClass->TestSingleClick(0, 8, 0);
  testClass->TestSingleClick(0, 10, 0);
  testClass->TestClickDrag(0, 1, 1, 2);
  testClass->TestClickDrag(0, 1, 3, 4);
  testClass->TestClickDrag(0, 1, 5, 6);
  testClass->TestClickDrag(0, 1, 7, 6); // Size of mouse stroke is outside image, so no more voxels.
  testClass->TestClickDrag(0, 2, 1, 8); // Cursor size 2, making 2 crosses, 1 pixel apart.
  testClass->TestClickDrag(0, 4, 1, 24);
  testClass->TestClickDrag(0, 6, 1, 38);
  testClass->TestClickDrag(0, 8, 1, 40);
  testClass->TestClickDrag(0, 10, 1, 40); // Size of cross goes outside image, so no more voxels.
  testClass->TestClickDrag(1, 10, 1, 40); // Test same in other image.
  testClass->TestErase();
  testClass->TestScan();
  testClass->TestClickJustOutOfBounds();
  testClass->TestClickWayOutOfBounds();

  delete testClass;
  MITK_TEST_END();
}


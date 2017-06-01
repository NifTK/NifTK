#include <iostream>
#include <fstream>
#include <typeinfo>

#include <mitkIOUtil.h>
#include <mitkSceneIO.h>
#include <niftkExtractDataFromMITKSceneCLP.h>
#include <itkNifTKImageIOFactory.h>

int main(int argc, char **argv)
{
   PARSE_ARGS;

   itk::NifTKImageIOFactory::Initialize();

   int returnStatus = EXIT_FAILURE;

   try
   {
      if ( input.length() != 0 )
      {
         // Read the specified MITK scene
         mitk::SceneIO::Pointer sceneIO = mitk::SceneIO::New();
         mitk::DataStorage::Pointer dataStorage = sceneIO->LoadScene(input);

         // Extract and save the point nodes
         mitk::DataStorage::SetOfObjects::ConstPointer nodes = dataStorage->GetAll();
         for (unsigned int i = 0; i < nodes->size(); i++)
         {
            if(datatype.compare("pointset")==0)
            {
               mitk::PointSet::Pointer ps = dynamic_cast<mitk::PointSet *>(nodes->at(i)->GetData());
               if (ps.IsNull())
               {
                  continue;
               }
               std::string nodeName = nodes->at(i)->GetName();
               std::cout << "Saving the point set named: " << nodeName << std::endl;

               mitk::IOUtil::Save(ps, output + "_" + nodeName + ".mps");
            }
            else if(datatype.compare("image")==0)
            {
               mitk::Image::Pointer img = dynamic_cast<mitk::Image *>(nodes->at(i)->GetData());
               if (img.IsNull())
               {
                  continue;
               }
               std::string nodeName = nodes->at(i)->GetName();
               std::cout << "Saving the image named: " << nodeName << std::endl;

               mitk::IOUtil::Save(img, output + "_" + nodeName + ".nii.gz");
            }
            else{
               MITK_ERROR << "Unsupported datatype: " << datatype;
               returnStatus = -3;

            }
         }
         returnStatus = EXIT_SUCCESS;
      }
   }
   catch (std::exception& e)
   {
     MITK_ERROR << "Caught std::exception:" << e.what();
     returnStatus = -1;
   }
   catch (...)
   {
     MITK_ERROR << "Caught unknown exception:";
     returnStatus = -2;
   }
   return returnStatus;
}

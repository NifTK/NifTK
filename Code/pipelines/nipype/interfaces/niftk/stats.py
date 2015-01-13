"""
   Interface for niftk statistical tools
"""

from niftk.base import NIFTKCommand, NIFTKCommandInputSpec, getNiftkPath
from nipype.interfaces.base import (TraitedSpec, File, traits)



class ComputeDiceScoreInputSpec(NIFTKCommandInputSpec):
    
    in_file1 = File(argstr="%s", exists=True, mandatory=True,
                        desc="First roi image")
    in_file2 = File(argstr="%s", exists=True, mandatory=True,
                        desc="Second roi image")
    
class ComputeDiceScoreOutputSpec(TraitedSpec):
    out_dict = traits.Dict(desc='Output array containing the dice score.\n'+
        'The first value is the label index and the second is the Dice score')


class ComputeDiceScore(NIFTKCommand):
    
    input_spec = ComputeDiceScoreInputSpec  
    output_spec = ComputeDiceScoreOutputSpec

    def _run_interface(self, runtime):
        roi_file1 = self.inputs.in_file1
        roi_file2 = self.inputs.in_file2
        self.out_dict=self.compute_dice_score(roi_file1, roi_file2)
        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_dict'] = self.out_dict
        return outputs
        

    def compute_dice_score(self,
                           roi_file1,
                           roi_file2):
                               
        # Reorient the input images to remove any nibabel reading error
        reorient1=fsl.Reorient2Std()
        reorient2=fsl.Reorient2Std()
        reorient1.inputs.in_file=roi_file1
        reorient2.inputs.in_file=roi_file2
        reoriented_filename1=reorient1.run().outputs.out_file
        reoriented_filename2=reorient2.run().outputs.out_file
        
        # Read the input images
        in_img1=nib.load(reoriented_filename1).get_data()
        in_img2=nib.load(reoriented_filename2).get_data()
        
        # Get the min and max label values
        min_label_value = np.int32(np.min([np.min(in_img1), np.min(in_img2)]))
        max_label_value = np.int32(np.max([np.max(in_img1), np.max(in_img2)]))

        # Iterate over all label values
        out_dict=dict()
        for l in range(min_label_value,max_label_value+1):
            mask1 = in_img1==l
            mask2 = in_img2==l
            mask3 = (in_img1==l) + (in_img2==l)
            if np.sum(mask1)+np.sum(mask2) != 0:
                out_dict[l]=0.5*(np.sum(mask1)+np.sum(mask2))/np.sum(mask3)
            
        return out_dict






class CalculateAffineDistancesInputSpec(NIFTKCommandInputSpec):
    transformation1_list = traits.List(traits.File(exists=True), mandatory=True, desc='List of affine transformations')
    transformation2_list = traits.List(traits.File(exists=True), mandatory=True, desc='List of affine transformations')
    
    
class CalculateAffineDistancesOutputSpec(TraitedSpec):
    out_array   = traits.Array(desc='Array of distances between the paired transformations')

class CalculateAffineDistances(NIFTKCommand):

    """

    Examples
    --------

    """
    input_spec = CalculateAffineDistancesInputSpec  
    output_spec = CalculateAffineDistancesOutputSpec

    def _run_interface(self, runtime):
        transformation1_list = self.inputs.transformation1_list
        transformation2_list = self.inputs.transformation2_list
        self.distances=self.calculate_distance_between_affines(transformation1_list, transformation2_list)
        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_array'] = self.distances
        return outputs
        

    def read_file_to_matrix(self, file_name):
        mat = np.zeros((4,4))
        with open(file_name, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ',skipinitialspace=True)
                row_ind = 0
                for row in reader:
                    #print row
                    for col_ind in range(4):
                        if row[col_ind] == ' ':
                            row.pop(col_ind)
                        mat[row_ind,col_ind] = row[col_ind]
                        #print row[col_ind]
                    row_ind = row_ind + 1
        return mat

    def calculate_distance_between_affines(self,list1_aff, list2_aff):
        distances = np.zeros((len(list1_aff),1))    
        
        for i in range(len(list1_aff)):
            # Read affine matrices
            file1 = list1_aff[i]
            file2 = list2_aff[i]
            mat1 = self.read_file_to_matrix(file1)
            mat2 = self.read_file_to_matrix(file2)
            log_mat1 = la.logm(mat1)
            log_mat2 = la.logm(mat2)
            distances[i,0] = ((log_mat2 - log_mat1)*(log_mat2 - log_mat1)).sum()
            
            
        return distances
        

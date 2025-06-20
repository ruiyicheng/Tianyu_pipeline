
import os
import sys
sys.path.append(os.path.abspath("/home/test/workspace/Tianyu_pipeline/algorithm/code"))
import utils.dataloader as dataloader
import pandas as pd
import numpy as np

class SelectKeyTargets:
    def __init__(self):
        self.data_loader = dataloader.dataloader()

    def select(self,sky_id,inst_id):
        """
        Select key targets from the sky based on the sky_id and inst_id.
        This function retrieves the target list for the given sky_id and inst_id,
        sorts them by their 'priority' column, and returns the top 'target_num' targets.
        """
        pd_target_list = self.data_loader.query_input_catalog(sky_id, inst_id)
        pd_target_list_with_dist = pd_target_list[(pd_target_list['parallax_over_error']>1)&(pd_target_list['parallax']>0.0001)]
        pd_target_list_with_dist['bprp'] = pd_target_list_with_dist['phot_bp_mean_mag'] - pd_target_list_with_dist['phot_rp_mean_mag']
        pd_target_list_with_dist['MG'] = pd_target_list_with_dist['phot_g_mean_mag'] + 5 * np.log10(pd_target_list_with_dist['parallax'] / 1000) + 5

        is_WD =  ((pd_target_list_with_dist['MG'])> 5.047*pd_target_list_with_dist['bprp'] + 5.93)&(pd_target_list_with_dist['MG']>5)&(pd_target_list_with_dist['MG']>6*pd_target_list_with_dist['bprp']**3-21.77*pd_target_list_with_dist['bprp']**2+27.91*pd_target_list_with_dist['bprp']+0.897)&(pd_target_list_with_dist['bprp']<1.7)&(pd_target_list_with_dist['phot_bp_mean_mag']>0.001)

        pd_WD_list = pd_target_list_with_dist[is_WD]
        
        is_RD = (pd_target_list_with_dist['MG']<6*pd_target_list_with_dist['bprp']**3-21.77*pd_target_list_with_dist['bprp']**2+27.91*pd_target_list_with_dist['bprp']+0.897)&(pd_target_list_with_dist['bprp']>1.4)
        pd_RD_list = pd_target_list_with_dist[is_RD]
        pd_bright_RD_list = pd_RD_list[(pd_RD_list['phot_g_mean_mag']<14)]
        print(f"Number of WD targets: {len(pd_WD_list)}, Number of bright RD targets: {len(pd_bright_RD_list)}")

        is_TNO = (pd_target_list_with_dist['MG']<6*pd_target_list_with_dist['bprp']**3-21.77*pd_target_list_with_dist['bprp']**2+27.91*pd_target_list_with_dist['bprp']+0.897)&(pd_target_list_with_dist['bprp']<0.75)&(pd_target_list_with_dist['phot_bp_mean_mag']>0.001)
        is_TNO = is_TNO&(pd_target_list_with_dist['phot_g_mean_mag']<14)&(pd_target_list_with_dist['phot_g_mean_mag']>10)
        pd_TNO_list = pd_target_list_with_dist[is_TNO]
        print(f"Number of TNO targets: {len(pd_TNO_list)}")

        print(f'total number of selected targets: {2/np.pi*(len(pd_TNO_list)+ len(pd_bright_RD_list) + len(pd_WD_list))}')

        


if __name__ == "__main__":
    sky_id = 2  # Example sky ID
    inst_id = 2  # Example instrument ID

    selector = SelectKeyTargets()
    selector.select(sky_id, inst_id)
import pytorch_lightning as pl
import torch 
from omegaconf import DictConfig, OmegaConf
from typing import Any
from modules.utils.fetch import fetch_data_module
import cv2
from Huang_predictor import Predictor
import hydra
from config.modifier import dynamically_modify_train_config
from pathlib import Path

@hydra.main(config_path='config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    '''
    adapt RVT-repo-validation.py for testing
    '''
    dynamically_modify_train_config(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')



    # adapted model with visualization ability
    ckpt_path = Path(config.checkpoint)
    model = Predictor(config)  
    model = model.load_from_checkpoint(str(ckpt_path), **{'full_config': config}) # method of pl.LightningModule


    # data
    data_module = fetch_data_module(config=config) # -> pl.LightningDataModule
    
    data_module.setup('test') # method of pl.LightningDataModule
    test_dataloader = data_module.test_dataloader() # method of pl.LightningDataModule

    list_merged_img = []
    with torch.inference_mode(): # in case of gradient calculation which occupies a lot of memories

        for batch_idx, batch in enumerate(test_dataloader):
            print('****** Take single batch from test_dataloader *******')
            outputs = model(batch=batch , batch_idx=batch_idx)
            list_merged_img = model.draw_bbox_on_ev_img(batch=batch, outputs=outputs)

            # TODO: why only 2 images? Change the batch_size.eval in visualize.sh
            for idx, img in enumerate(list_merged_img):
                
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # window_name = f"Image {idx}: {img_bgr[idx]}"
                
                print(f'## showing the {idx} img ##')
                cv2.imshow('result', img_bgr) # static name
                cv2.waitKey(1000)

                # cv2.destroyWindow(window_name)

if __name__ == '__main__':
    main()
        

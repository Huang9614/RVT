import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from typing import Any


# prediction
from data.genx_utils.labels import ObjectLabels
from data.utils.types import DataType, ObjDetOutput
from models.detection.yolox.utils.boxes import postprocess
from models.detection.yolox_extension.models.detector import YoloXDetector
from utils.evaluation.prophesee.io.box_loading import to_prophesee
from utils.padding import InputPadderFromShape
from modules.utils.detection import BackboneFeatureSelector, EventReprSelector, RNNStates, Mode, mode_2_string, merge_mixed_batches


# ev_with_bbox
from utils.evaluation.prophesee.visualize.vis_utils import LABELMAP_GEN1, draw_bboxes
import cv2
import numpy as np

# draw_bbox_on_ev_img
from einops import rearrange, reduce

import time

class Predictor(pl.LightningModule):
    '''
        From the functional perspective, this class works similar with class Module: 1. inherent from pl.LightningModule; 2. consume data and output predictions;

        Even though this class inherents from pl.LightningModule, 
        I didn't use any other "hookers" of it, listed here [https://lightning.ai/docs/pytorch/stable/common/lightning_module.html]
        
        - def forward           :   generate the predictions in the form of TODO
        - def ev_repr_to_img    :   convert event stream to images
        - def ev_img_with_bbox  :   generate the event_frame with predicted bboxes und corresponding labels
    '''

    def __init__(self, full_config:DictConfig):
        super().__init__()

        self.full_config = full_config

        self.mdl_config = full_config.model
        
        in_res_hw = tuple(self.mdl_config.backbone.in_res_hw)
        
        self.input_padder = InputPadderFromShape(desired_hw=in_res_hw)

        self.mdl = YoloXDetector(self.mdl_config)

        self.mode_2_rnn_states: Dict[Mode, RNNStates] = {
            Mode.TRAIN: RNNStates(),
            Mode.VAL: RNNStates(),
            Mode.TEST: RNNStates(),
        }

    def forward(self, batch: Any, batch_idx:int):
        '''
            Follow the idea from training_step in the class Module(pl.LightningModule) in detection.py;

            read dataset from dataloader and return structured output through YoloxDetector;

            - batch: 
        '''

        # start_time = time.time()    # just for measuring the inference time

        batch = merge_mixed_batches(batch)
        data = batch['data']
        worker_id = batch['worker_id']


        mode = Mode.TEST
        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        token_mask_sequence = data.get(DataType.TOKEN_MASK, None) # method of built-in datatype: Dictionary, return the corresoponding value 


        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample) # method of instance of class RNNStates


        sequence_len = len(ev_tensor_sequence)
        print(f"-- the ev_tensor_sequence is {sequence_len}, output from Huang_predictor.py line 85. --\n")
        batch_size = len(sparse_obj_labels[0])
        print(f"-- And the batch_size is {batch_size}. --\n")

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        backbone_feature_selector = BackboneFeatureSelector()
        ev_repr_selector = EventReprSelector()

        start = time.time()
        print(f" start the for loop for event_seq\n")
        obj_labels = list()
        
        for tidx in range(sequence_len):
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            if token_mask_sequence is not None:
                token_masks = self.input_padder.pad_token_mask(token_mask=token_mask_sequence[tidx])
            else:
                token_masks = None

            backbone_features, states = self.mdl.forward_backbone(x=ev_tensors,
                                                                  previous_states=prev_states,
                                                                  token_mask=token_masks)


            prev_states = states


            current_labels, valid_batch_indices = sparse_obj_labels[tidx].get_valid_labels_and_batch_indices() # method of class SparselyBatchedObjectLabels from labels.py
            if len(current_labels) > 0:
                backbone_feature_selector.add_backbone_features(backbone_features=backbone_features,
                                                                selected_indices=valid_batch_indices)
                obj_labels.extend(current_labels)
                ev_repr_selector.add_event_representations(event_representations=ev_tensors,
                                                           selected_indices=valid_batch_indices)

        end = time.time()
        duration = end - start
        print(f"end for loop.\n The for loop takes {duration} secs.\n")

        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)
        

        selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()
        labels_yolox = ObjectLabels.get_labels_as_batched_tensor(obj_label_list=obj_labels, format_='yolox')
        labels_yolox = labels_yolox.to(dtype=self.dtype)

        predictions, losses = self.mdl.forward_detect(backbone_features=selected_backbone_features,
                                                      targets=labels_yolox)


        pred_processed = postprocess(prediction=predictions,
                                     num_classes=self.mdl_config.head.num_classes,
                                     conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                     nms_thre=self.mdl_config.postprocess.nms_threshold)

        loaded_labels_proph, yolox_preds_proph = to_prophesee(obj_labels, pred_processed)


        predictions = predictions[-batch_size:]
        obj_labels = obj_labels[-batch_size:]

        
        output = {
                ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-batch_size:],
                ObjDetOutput.PRED_PROPH: yolox_preds_proph[-batch_size:],
                ObjDetOutput.EV_REPR: ev_repr_selector.get_event_representations_as_list(start_idx=-batch_size),
                ObjDetOutput.SKIP_VIZ: False,
                'loss': losses['loss']
            }

    
        # end_time = time.time()

        # duration = end_time - start_time

        # print(f' --- It takes {duration} seconds for the inference ---')
        return output


    def ev_repr_to_img(self, x: np.ndarray):
        '''
            directly copy from class VizCallbackBase;

            - img: [h,w,3] RGB image
        '''
        
        ch, ht, wd = x.shape[-3:]
        assert ch > 1 and ch % 2 == 0
        ev_repr_reshaped = rearrange(x, '(posneg C) H W -> posneg C H W', posneg=2)
        img_neg = np.asarray(reduce(ev_repr_reshaped[0], 'C H W -> H W', 'sum'), dtype='int32')
        img_pos = np.asarray(reduce(ev_repr_reshaped[1], 'C H W -> H W', 'sum'), dtype='int32')
        img_diff = img_pos - img_neg
        img = 127 * np.ones((ht, wd, 3), dtype=np.uint8)
        img[img_diff > 0] = 255
        img[img_diff < 0] = 0

        return img


    def draw_bbox_on_ev_img(self, batch: Any, outputs: Any):
        '''
            follow the idea of function on_train_batch_end_custom in class DetectionVizCallback

            generate event images with predicted boungding boxes
        '''
        print(" -- start to drawing bbox -- \n")
        start_time  = time.time()

        ev_tensors = outputs[ObjDetOutput.EV_REPR]
        num_samples = len(ev_tensors)

        merged_img = []
        for sample_idx in range(num_samples):
            ev_img = self.ev_repr_to_img(ev_tensors[sample_idx].cpu().numpy())

            predictions_proph = outputs[ObjDetOutput.PRED_PROPH][sample_idx]
            prediction_img = ev_img.copy()
            draw_bboxes(prediction_img, predictions_proph, labelmap=LABELMAP_GEN1)

            labels_proph = outputs[ObjDetOutput.LABELS_PROPH][sample_idx]
            label_img = ev_img.copy()
            draw_bboxes(label_img, labels_proph, labelmap=LABELMAP_GEN1)

            merged_img.append(rearrange([prediction_img, label_img], 'pl H W C -> (pl H) W C', pl=2, C=3)) # pl = prediction_img + label_img
        
        end_time = time.time()
        duration = end_time - start_time

        print(f" -- Drawing Bbox done, takes {duration} seconds. \n")
        return merged_img
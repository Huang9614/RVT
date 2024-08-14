# Core Implementations

## class PartitionAttentionCl(nn.Module)
- LayerNorm + Block-SA/Grid-SA + LayerScale + PathDrop + LayerNorm + MLP + LayerScale + PathDrop btw. residul connection in RVT Block
- [Defined](https://github.com/uzh-rpg/RVT/blob/master/models/layers/maxvit/maxvit.py#L242)
- [Used](https://github.com/uzh-rpg/RVT/blob/master/models/detection/recurrent_backbone/maxvit_rnn.py#L180)
- 
## LSTM
- lstm in RVT block
- [defined](https://github.com/uzh-rpg/RVT/blob/master/models/layers/rnn.py)
- [used](https://github.com/uzh-rpg/RVT/blob/master/models/detection/recurrent_backbone/maxvit_rnn.py#L152)

## class RNNDetectorStage(nn.Module)
- logic of RVT block
- [defined and used](https://github.com/uzh-rpg/RVT/blob/master/models/detection/recurrent_backbone/maxvit_rnn.py#L66)

## class RNNDetector(BaseDetector)
- output after 4 RVT blocks
- [defined](https://github.com/uzh-rpg/RVT/blob/master/models/detection/recurrent_backbone/maxvit_rnn.py#L66)
- [used](https://github.com/uzh-rpg/RVT/tree/master/models/detection/recurrent_backbone))
  - logic of backbone

## class YoloXDetector(th.nn.Module):
- accept the output from 4 RVT block and use Yolox for object detection
- [defined](https://github.com/uzh-rpg/RVT/blob/master/models/detection/yolox_extension/models/detector.py#L18)
- [used](https://github.com/uzh-rpg/RVT/blob/master/modules/detection.py#L33)
  - finally used in `train.py` and `validation.py` files


# Extention: Visulization

## idea
1. instantiate the model, called Predictor, load parameters from checkpoint
2. instantiate test_dataloader
3. feed each batch from the test_dataloader into the model to get the predicted results, which consist of LABELS_PROPH, PRED_PROPH, EV_ERPR and SKIP_VIZ
4. feed the results into draw_bbox_on_ev_img, to get the image with bbox and predicted labels
5. using cv2 to show the merged




# Method of dealing with Event Stream 





# Notes

## configure files


# others
```bash
tks-huston.fzi.de:3000 # check the option All

nvidia-smi # check what kind of programs are running on the current gpu

ssh ids-imperator # connect to other workstation, name from All
```


## [tmux tutorial](https://hamvocke.com/blog/a-quick-and-easy-guide-to-tmux)







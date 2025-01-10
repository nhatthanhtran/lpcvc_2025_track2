# CUDA_VISIBLE_DEVICES=1 mpirun -n 1 python entry.py evaluate \
#             --conf_files configs/xdecoder/focalt_unicl_lang_finetune.yaml \
#             --overrides \
#             COCO.INPUT.IMAGE_SIZE 1024 \
#             MODEL.DECODER.CAPTIONING.ENABLED False \
#             MODEL.DECODER.RETRIEVAL.ENABLED False \
#             MODEL.DECODER.GROUNDING.ENABLED True \
#             COCO.TEST.BATCH_SIZE_TOTAL 1 \
#             COCO.TRAIN.BATCH_SIZE_TOTAL 1 \
#             COCO.TRAIN.BATCH_SIZE_PER_GPU 1 \
#             VLP.TEST.BATCH_SIZE_TOTAL 1 \
#             VLP.TRAIN.BATCH_SIZE_TOTAL 1 \
#             VLP.TRAIN.BATCH_SIZE_PER_GPU 1 \
#             MODEL.DECODER.HIDDEN_DIM 512 \
#             MODEL.ENCODER.CONVS_DIM 512 \
#             MODEL.ENCODER.MASK_DIM 512 \
#             ADE20K.TEST.BATCH_SIZE_TOTAL 1 \
#             FP16 True \
#             WEIGHT True \
#             RESUME_FROM data/output/test/focalt_unicl_lang_finetune.yaml_conf~/run_1/00197015/default/model_state_dict.pt

# CUDA_VISIBLE_DEVICES=0 mpirun -n 1 python entry.py train \
#             --conf_files configs/xdecoder/focalt_unicl_lang_test.yaml \
#             --overrides \
#             FP16 True \
#             COCO.INPUT.IMAGE_SIZE 1024 \
#             MODEL.DECODER.HIDDEN_DIM 512 \
#             MODEL.ENCODER.CONVS_DIM 512 \
#             MODEL.ENCODER.MASK_DIM 512 \
#             MODEL.DECODER.CAPTIONING.ENABLED False \
#             MODEL.DECODER.RETRIEVAL.ENABLED False \
#             MODEL.DECODER.GROUNDING.ENABLED True \
#             MODEL.DECODER.CAPTIONING_WEIGHT 8 \
#             MODEL.DECODER.RETRIEVAL_WEIGHT 8 \
#             MODEL.DECODER.TOP_CAPTIONING_LAYERS 3 \
#             MODEL.DECODER.TOP_RETRIEVAL_LAYERS 3 \
#             MODEL.DECODER.TOP_GROUNDING_LAYERS 6 \
#             MODEL.DECODER.GROUNDING.TEXT_WEIGHT 2.0 \
#             MODEL.DECODER.GROUNDING.CLASS_WEIGHT 0.5 \
#             COCO.TEST.BATCH_SIZE_TOTAL 2 \
#             COCO.TRAIN.BATCH_SIZE_TOTAL 2 \
#             COCO.TRAIN.BATCH_SIZE_PER_GPU 2 \
#             VLP.TEST.BATCH_SIZE_TOTAL 2 \
#             VLP.TRAIN.BATCH_SIZE_TOTAL 8 \
#             VLP.TRAIN.BATCH_SIZE_PER_GPU 8 \
#             VLP.DATALOADER.NUM_WORKERS 16 \
#             ADE20K.TEST.BATCH_SIZE_TOTAL 2 \
#             REF.TEST.BATCH_SIZE_TOTAL 2 \
#             SOLVER.LR_MULTIPLIER.lang_encoder 0.1 \
#             WEIGHT True \
#             RESUME_FROM ./ckpts/focalt_in21k_yfcc_gcc_xdecoder_unicl.pt

# CUDA_VISIBLE_DEVICES=0,1 nohup nohup mpirun -n 2 python entry.py train \
#             --conf_files configs/xdecoder/focalt_unicl_lang_test.yaml \
#             --overrides \
#             FP16 True \
#             COCO.INPUT.IMAGE_SIZE 1024 \
#             MODEL.DECODER.HIDDEN_DIM 512 \
#             MODEL.ENCODER.CONVS_DIM 512 \
#             MODEL.ENCODER.MASK_DIM 512 \
#             MODEL.DECODER.CAPTIONING.ENABLED False \
#             MODEL.DECODER.RETRIEVAL.ENABLED False \
#             MODEL.DECODER.GROUNDING.ENABLED True \
#             MODEL.DECODER.CAPTIONING_WEIGHT 8 \
#             MODEL.DECODER.RETRIEVAL_WEIGHT 8 \
#             MODEL.DECODER.TOP_CAPTIONING_LAYERS 3 \
#             MODEL.DECODER.TOP_RETRIEVAL_LAYERS 3 \
#             MODEL.DECODER.TOP_GROUNDING_LAYERS 6 \
#             MODEL.DECODER.GROUNDING.TEXT_WEIGHT 2.0 \
#             MODEL.DECODER.GROUNDING.CLASS_WEIGHT 0.5 \
#             COCO.TEST.BATCH_SIZE_TOTAL 4 \
#             COCO.TRAIN.BATCH_SIZE_TOTAL 4 \
#             COCO.TRAIN.BATCH_SIZE_PER_GPU 2 \
#             VLP.TEST.BATCH_SIZE_TOTAL 2 \
#             VLP.TRAIN.BATCH_SIZE_TOTAL 16 \
#             VLP.TRAIN.BATCH_SIZE_PER_GPU 8 \
#             VLP.DATALOADER.NUM_WORKERS 16 \
#             ADE20K.TEST.BATCH_SIZE_TOTAL 2 \
#             REF.TEST.BATCH_SIZE_TOTAL 2 \
#             SOLVER.LR_MULTIPLIER.lang_encoder 0.1 \
#             WEIGHT True \
#             RESUME_FROM ./ckpts/focalt_in21k_yfcc_gcc_xdecoder_unicl.pt > train.txt &

# CUDA_VISIBLE_DEVICES=0,1 nohup nohup mpirun -n 2 python entry.py train \
#             --conf_files configs/xdecoder/focalt_unicl_lang_test.yaml \
#             --overrides \
#             FP16 True \
#             COCO.INPUT.IMAGE_SIZE 1024 \
#             MODEL.DECODER.HIDDEN_DIM 512 \
#             MODEL.ENCODER.CONVS_DIM 512 \
#             MODEL.ENCODER.MASK_DIM 512 \
#             MODEL.DECODER.CAPTIONING.ENABLED False \
#             MODEL.DECODER.RETRIEVAL.ENABLED False \
#             MODEL.DECODER.GROUNDING.ENABLED True \
#             MODEL.DECODER.CAPTIONING_WEIGHT 8 \
#             MODEL.DECODER.RETRIEVAL_WEIGHT 8 \
#             MODEL.DECODER.TOP_CAPTIONING_LAYERS 3 \
#             MODEL.DECODER.TOP_RETRIEVAL_LAYERS 3 \
#             MODEL.DECODER.TOP_GROUNDING_LAYERS 6 \
#             MODEL.DECODER.GROUNDING.TEXT_WEIGHT 2.0 \
#             MODEL.DECODER.GROUNDING.CLASS_WEIGHT 0.5 \
#             COCO.TEST.BATCH_SIZE_TOTAL 4 \
#             COCO.TRAIN.BATCH_SIZE_TOTAL 4 \
#             COCO.TRAIN.BATCH_SIZE_PER_GPU 2 \
#             VLP.TEST.BATCH_SIZE_TOTAL 4 \
#             VLP.TRAIN.BATCH_SIZE_TOTAL 256 \
#             VLP.TRAIN.BATCH_SIZE_PER_GPU 32 \
#             VLP.DATALOADER.NUM_WORKERS 32 \
#             ADE20K.TEST.BATCH_SIZE_TOTAL 4 \
#             REF.TEST.BATCH_SIZE_TOTAL 2 \
#             SOLVER.LR_MULTIPLIER.lang_encoder 0.1 \
#             WEIGHT True \
#             RESUME_FROM ./ckpts/focalt_in21k_yfcc_gcc_xdecoder_unicl.pt > train.txt &

CUDA_VISIBLE_DEVICES=0,1 nohup nohup mpirun -n 2 python entry.py train \
            --conf_files configs/xdecoder/focalt_unicl_lang_finetune.yaml \
            --overrides \
            FP16 True \
            COCO.INPUT.IMAGE_SIZE 1024 \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            MODEL.DECODER.CAPTIONING.ENABLED False \
            MODEL.DECODER.RETRIEVAL.ENABLED False \
            MODEL.DECODER.GROUNDING.ENABLED True \
            MODEL.DECODER.CAPTIONING_WEIGHT 8 \
            MODEL.DECODER.RETRIEVAL_WEIGHT 8 \
            MODEL.DECODER.TOP_CAPTIONING_LAYERS 3 \
            MODEL.DECODER.TOP_RETRIEVAL_LAYERS 3 \
            MODEL.DECODER.TOP_GROUNDING_LAYERS 6 \
            MODEL.DECODER.GROUNDING.TEXT_WEIGHT 2.0 \
            MODEL.DECODER.GROUNDING.CLASS_WEIGHT 0.5 \
            COCO.TEST.BATCH_SIZE_TOTAL 4 \
            COCO.TRAIN.BATCH_SIZE_TOTAL 4 \
            COCO.TRAIN.BATCH_SIZE_PER_GPU 2 \
            VLP.TEST.BATCH_SIZE_TOTAL 2 \
            VLP.TRAIN.BATCH_SIZE_TOTAL 16 \
            VLP.TRAIN.BATCH_SIZE_PER_GPU 8 \
            VLP.DATALOADER.NUM_WORKERS 16 \
            ADE20K.TEST.BATCH_SIZE_TOTAL 2 \
            REF.TEST.BATCH_SIZE_TOTAL 2 \
            SOLVER.LR_MULTIPLIER.lang_encoder 0.1 \
            WEIGHT True \
            RESUME_FROM data/output/test/focalt_unicl_lang_finetune.yaml_conf~/run_1/00197015/default/model_state_dict.pt > train_finetune_qnn.txt &
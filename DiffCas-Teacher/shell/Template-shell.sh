#train
python3 main.py --config DiffCas-Reformat/DiffCas-Teacher/configs/Template-DiffCas-T.yaml --train --sample_at_start --save_top --gpu_ids 0 \
--resume_model path/to/model_ckpt --resume_optim path/to/optim_ckpt
#test
python3 main.py --config DiffCas-Reformat/DiffCas-Teacher/configs/Template-DiffCas-T.yaml --sample_to_eval --gpu_ids 0 \
--resume_model path/to/model_ckpt --resume_optim path/to/optim_ckpt
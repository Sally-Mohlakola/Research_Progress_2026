python gather_rdm.py     --checkpoint_name run_03    --diamond_name round_diamond_gia     --batch_size 9000000    --num_batches 100     --theta_bins 8     --phi_bins 16     --max_depth 64

 python eval.py     --checkpoint_name run_06    --diamond_name round_brilliant_sharp_culet     --spp 8    --width 768     --height 768     --output renders/diamond_test_11

python train_models.py     --checkpoint_name run_03     --diamond_name round_diamond_gia     --epochs_m 5000     --epochs_t 3000     --lr_m 0.001     --lr_t 0.001     --batch_size 4096     --no_plot

=======================================
python gather_rdm.py     --checkpoint_name run_01    --diamond_name round_diamond_gia     --batch_size 100000     --num_batches 10     --theta_bins 8     --phi_bins 16     --max_depth 64
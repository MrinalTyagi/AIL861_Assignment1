
python generate.py --checkpoint_path output_context_64/checkpoints/best_checkpoint_50.pth \
    --temperature 0.7 \
    --beam_width 5 \
    --decoding beam_search \
    --num_samples 100 \
    --experiment_name beam_search_5

python generate.py --checkpoint_path output_context_64/checkpoints/best_checkpoint_50.pth \
    --temperature 0.7 \
    --beam_width 10 \
    --decoding beam_search \
    --num_samples 100 \
    --experiment_name beam_search_10

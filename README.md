# Reiterative Domain Aware Multi-Target Adaptation
## Paper link: https://link.springer.com/chapter/10.1007/978-3-031-16788-1_5

### Sample commands
**Office-Home (Source: Art)** <br/>
$ python code/main.py --method 'CDAN' --seed 0 --reiterationNumber 10 --source_batch 48 --target_batch 16 --encoder ViT --dataset office-home --data_root ./office-home/ --source art --target clipart product real --source_iters 500 --adapt_iters 1000 --finetune_iters 15000 --lambda_node 0.3 --num_workers 4 --output_dir officeHome_Reit_SBatch48TBatch16/art_restTry1 

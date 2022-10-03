# Reiterative Domain Aware Multi-Target Adaptation
## Paper link: https://link.springer.com/chapter/10.1007/978-3-031-16788-1_5

### Sample commands
**Pay attention to parameters like data_root, output_dir** <br/>
**Office-Home (Source: Art)** <br/>
$ python code/main.py --method 'CDAN' --seed 0 --reiterationNumber 10 --source_batch 48 --target_batch 16 --encoder ViT --dataset office-home --data_root ./office-home/ --source art --target clipart product real --source_iters 500 --adapt_iters 1000 --finetune_iters 15000 --lambda_node 0.3 --num_workers 4 --output_dir officeHome_Reit_SBatch48TBatch16/art_restTry1 

**Office-Home (Source: Clipart)** <br/>
$ python code/main.py --method 'CDAN' --seed 0 --reiterationNumber 10 --source_batch 48 --target_batch 16 --encoder ViT --dataset office-home --data_root ./office-home/ --source clipart --target art product real --source_iters 500 --adapt_iters 1000 --finetune_iters 15000 --lambda_node 0.3 --num_workers 4 --output_dir officeHome_Reit_SBatch48TBatch16/clipart_restTry1

**Office-Home (Source: Product)** <br/>
$ python code/main.py --method 'CDAN' --seed 0 --reiterationNumber 10 --source_batch 48 --target_batch 16 --encoder ViT --dataset office-home --data_root ./office-home/ --source product --target art clipart real --source_iters 500 --adapt_iters 1000 --finetune_iters 15000 --lambda_node 0.3 --num_workers 4 --output_dir officeHome_Reit_SBatch48TBatch16/product_restTry1

**Office-Home (Source: Real)** <br/>
$ python code/main.py --method 'CDAN' --seed 0 --reiterationNumber 10 --source_batch 48 --target_batch 16 --encoder ViT --dataset office-home --data_root ./office-home/ --source real --target art clipart product --source_iters 500 --adapt_iters 1000 --finetune_iters 15000 --lambda_node 0.3 --num_workers 4 --output_dir officeHome_Reit_SBatch48TBatch16/real_restTry1

**Office-31 (Source: Webcam)** <br/>
$ python code/main.py --method 'CDAN' --seed 0 --reiterationNumber 3 --source_batch 48 --target_batch 16 --encoder ViT --dataset office31 --data_root ./office31/ --source webcam --target dslr amazon --source_iters 200 --adapt_iters 1000 --finetune_iters 15000 --lambda_node 0.3 --output_dir office31_Reit_SBatch48TBatch16/webcam_restTry1

**Office-31 (Source: Dslr)** <br/>
$ python code/main.py --method 'CDAN' --seed 0 --reiterationNumber 3 --source_batch 48 --target_batch 16 --encoder ViT --dataset office31 --data_root ./office31/ --source dslr --target webcam amazon --source_iters 200 --adapt_iters 1000 --finetune_iters 15000 --lambda_node 0.3 --output_dir office31_Reit_SBatch48TBatch16/dslr_restTry1

**Office-31 (Source: Amazon)** <br/>
$ python code/main.py --method 'CDAN' --seed 0 --reiterationNumber 3 --source_batch 48 --target_batch 16 --encoder ViT --dataset office31 --data_root ./office31/ --source amazon --target webcam dslr --source_iters 200 --adapt_iters 1000 --finetune_iters 15000 --lambda_node 0.3 --output_dir office31_Reit_SBatch48TBatch16/amazon_restTry1

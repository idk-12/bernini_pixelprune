你可以用脚本/data/lijie/PixelPrune/scripts/compare_runs.py 跑用了pixelprune和不用pixelprune（baseline）的结果对比，只需要改开头的BASELINE_DIR 和 PIXELPRUNE_DIR


dense模型 ： /data/lijie/PixelPrune/eval/outputs/full_baseline_qwen36/Qwen3.5-HF/T20260509-143825/TextVQA_VAL    vs    /data/lijie/PixelPrune/eval/outputs/pixelprune_doc_qwen36/Qwen3.5-HF/T20260510-145904/t=0.1-PIXELPRUNE_ANCHORED=false-TextVQA_VAL
结果：
Original	85.57 	901.3 	-	1.00 
Qwen3.6-27B-PixelPrune τ=0	85.58 	970.8 	-7.00%	0.99 
Qwen3.6-27B-PixelPrune τ=0.1	84.87 	861.1 	4.70%	0.78 

moe模型：  /data/lijie/PixelPrune/eval/outputs/full_baseline_qwen36_moe/Qwen3.5-HF/T20260604-015930/TextVQA_VAL       vs    /data/lijie/PixelPrune/eval/outputs/pixelprune_doc_qwen36_moe/Qwen3.5-HF/T20260606-211531/TextVQA_VAL  
结果：
Original	85.37 	605.7 	-	1.00 
Qwen3.6-35Bmoe-PixelPrune τ=0	85.20 	762.1 	-20.50%	0.99 
Qwen3.6-35Bmoe-PixelPrune τ=0.1	85.27 	743.8 	-18.60%	0.78

仅仅是换了一个moe模型： 1. 为什么都是baseline，moe的ttft时间开销减少了，怎么解释  2. 为什么同样是超参 PIXELPRUNE_THRESHOLD=0.1 PIXELPRUNE_ANCHORED=false ，  dense模型可以有4.7%的加速，但moe模型没有
    3. overhead主要来自pixel prune的计算， 为什么dense模型，在τ=0 overhead差不多等于70s，而moe模型的overhead有160s？

根据结果分析原因


/mnt/nfs/data/pretrained_models/Qwen3.5-9B/    /mnt/nfs/data/pretrained_models/Qwen3.6-27B/  /mnt/nfs/data/pretrained_models/Qwen3.6-35B-A3B/
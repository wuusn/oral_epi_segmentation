#python testTMA.py /ssd/Oral/train@256Overlap50/save/helloworld/checkpoints/best_dice.pth /raid10/_datasets/Oral/TMA/IHCHE/tissue_epi_0.3 _real_B.png '/raid10/_datasets/Oral/TMA/IHCHE/retrainOverlap50Result2'
#python testTMANorm.py /ssd/Oral/train@256Overlap50/save/norm/checkpoints/best_dice.pth /raid10/_datasets/Oral/TMA/IHCHE/tissue_epi_0.3 _real_B.png '/raid10/_datasets/Oral/TMA/IHCHE/retrainOverlap50NormResult2'
#python testTMA.py /ssd/Oral/train@256Random40/save/random40/checkpoints/best_dice.pth /raid10/_datasets/Oral/TMA/IHCHE/tissue_epi_0.3 _real_B.png '/raid10/_datasets/Oral/TMA/IHCHE/retrainRandom40Result'
python testTMA.py /ssd/Oral/train@256Random10Correct/save/random10/checkpoints/best_dice.pth /raid10/_datasets/Oral/TMA/IHCHE/tissue_epi_0.3 _real_B.png '/raid10/_datasets/Oral/TMA/IHCHE/retrainRandom10Result'


# Train 5-way, 1-shot miniimagent task with and without memory
optirun python k_shot.py --dataset miniimagenet --K 1 --N 5 --ep_len 2 --use_mem --task train
optirun python k_shot.py --dataset miniimagenet --K 1 --N 5 --ep_len 2 --task train

# Test 5-way, 1-shot miniimagent task with and without memory
optirun python k_shot.py --dataset miniimagenet --K 1 --N 5 --ep_len 2 --use_mem --task test
optirun python k_shot.py --dataset miniimagenet --K 1 --N 5 --ep_len 2 --task test



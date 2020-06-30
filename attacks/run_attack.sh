
#Experimental Setting
INPUT_DIR=../data/test/clean/fake/ 
OUTPUT_DIR=../data/test/advData/fake/
MODEL_NUM=1
MAX_EPS1=2
MAX_EPS2=6
MAX_EPS3=6
NUM_ITER=10
GPU_ID=0

python attack.py  --input_dir=${INPUT_DIR} --output_dir=${OUTPUT_DIR} \
                        --max_epsilon1=${MAX_EPS1} --max_epsilon2=${MAX_EPS2} --max_epsilon3=${MAX_EPS3} \
                        --model_num=${MODEL_NUM} --num_iter=${NUM_ITER} \
                        --gpu_id=${GPU_ID} 

#python eval.py  --imgs_dir=${OUTPUT_DIR} \
#                --max_epsilon1=${MAX_EPS1} --max_epsilon2=${MAX_EPS2} --max_epsilon3=${MAX_EPS3} \
#                --model_num=${MODEL_NUM} --num_iter=${NUM_ITER} \
#                --gpu_id=${GPU_ID}

conda env create -f environment.yml
conda activate test-sj


CSV_FILE=./dataset/submission.csv
INPUT_DIR=./dataset/valid
PREPROCESS_DIR=./dataset/preprocess
CHECKPOINT_PATH=./check


python pre_process.py --csv_file ${CSV_FILE} --input_folder ${INPUT_DIR} --output_folder ${PREPROCESS_DIR}

python test.py --csv_file ${CSV_FILE} --numpy_folder ${PREPROCESS_DIR} --checkpoint_path ${CHECKPOINT_PATH}

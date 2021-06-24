import os
import tensorflow as tf

from dngan import metrics, losses
import utils


def degrade_filter(k): return k.startswith('gauss')
def dataset_filter(k): return k in ['bsds500', 'set5', 'set14']


DEBUG = 'DNGAN_DEBUG' in os.environ
if DEBUG:
    print('-----------------------------------')
    print('WARNING: Debug configuration active')
    print('-----------------------------------')

# Use values from env variables
models_path = utils.get_from_environ('DNGAN_MODELS_PATH', 'models')
results_file = utils.get_from_environ('DNGAN_RESULTS_FILE', 'evaluation.csv')

# Evaluation metrics
metrics_dict = {
    'psnr': metrics.psnr,
    'ssim': metrics.ssim,
    # 'sm-ssim': metrics.ms_ssim,
    'vgg22': losses.vgg19_loss(2, 2),
    'vgg54': losses.vgg19_loss(5, 4),
    'vgg22-ba': losses.vgg19_ba_loss(2, 2),
    'vgg54-ba': losses.vgg19_ba_loss(5, 4),
    'fsim': metrics.fsim,
}
metrics_names = metrics_dict.keys()

# Get the datasets
config = {'img_size': [96, 96]} if DEBUG else {}
datasets_no_noise_map = utils.get_test_datasets(
    config=config, debug=DEBUG,
    dataset_filter=dataset_filter, degrade_filter=degrade_filter)
datasets_noise_map = utils.get_test_datasets(
    config={**config, 'degrade': {'type': 'gaussian-map'}}, debug=DEBUG,
    dataset_filter=dataset_filter, degrade_filter=degrade_filter)

# Get the models
model_files = [os.path.join(models_path, f) for f in os.listdir(models_path)]
model_files = list(filter(lambda m: os.path.isfile(m) and m.endswith('.h5'), model_files))

# Prepare the CSV
eval_csv = open(results_file, 'w')
eval_csv.write('Model,Dataset,Image_Id,')
eval_csv.write(','.join(metrics_names))
eval_csv.write('\n')

# Loop over the models
for model_idx, model_file in enumerate(model_files):
    model_name = model_file.split(os.path.sep)[-1][:-3]
    print(f"Evaluating model {model_name}...")

    # Load the model
    model = tf.keras.models.load_model(model_file, compile=False)

    # Use the datasets with an input noise map if the input has 4 channels
    if model.input.shape[-1] == 4:
        datasets = datasets_noise_map
    else:
        datasets = datasets_no_noise_map

    # Loop over all datasets
    for dataset_name, dataset in datasets.items():
        print(f"On dataset variant {dataset_name}...")

        # Loop over the dataset
        image_id = 0
        for x, y in dataset.batch(1):
            y_hat = tf.identity(model.predict(x))

            # Evaluate each metric
            res = [metrics_dict[m](y, y_hat) for m in metrics_names]

            # Write the result to the csv
            res_str = [str(r.numpy()) for r in res]
            eval_csv.write(f'{model_name},{dataset_name},{image_id},')
            eval_csv.write(','.join(res_str))
            eval_csv.write('\n')

            # Update image_id
            image_id += 1

    print(f"Finished {model_idx + 1}/{len(model_files)}")
    print()

eval_csv.close()

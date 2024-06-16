** Our dataset is not publicly availabe due to the issue of ethical and privacy concerns **

# sleep-abnormal-person-detection
Abnormal sleep patterns are often indicative of various diseases, making early detection crucial for timely intervention. The prolif- eration of wearable devices facilitates the continuous collection of physiological data, providing valuable insights into sleep health. This study proposes a novel deep learning model based on the Transformer architecture, specifically designed to detect abnormal sleep patterns using multivariate time series data. By integrating sleep-specific encoding, which incorporates the cyclic nature of sleep and temporal dependencies, our model enhances the detec- tion accuracy of sleep abnormalities. Evaluations conducted on a comprehensive dataset, collected from 714 patients using wearable devices, demonstrate that our model significantly outperforms tra- ditional methods in accuracy, precision, recall, and AUROC. The results underscore the potential of our approach in providing reli- able sleep health insights, thereby aiding in the early diagnosis and management of sleep-related disorders

## How to run the code from the terminal
### 1. Pre-configuration
- Start the terminal at the folder containing the entire codebase. Ex)
![image](https://github.com/Heej99/sleep-abnormal-person-unsupervised-detection/assets/42797013/f27b9b91-dc9b-4c88-9a3a-104cc6cde866)
- Make `experiments`, `finetuned`, `test_result` folder.
  - If you download it as is, there should be a folder. However, if it doesn't exist, you need to create it.
![image](https://github.com/Heej99/sleep-abnormal-person-unsupervised-detection/assets/42797013/6c4498b1-42b6-4090-827a-25295447b14b)

### 2. Pretraining(Unsupervised Learning Through Input Masking)
```bash
python src/main.py --output_dir experiments --comment "Pretraining_Sleep_Data" --name Sleep_pretrained --data_dir ./SleepData/ --data_class sleep --pattern TRAIN --val_ratio 0.3 --epochs 300 --lr 0.001 --optimizer RAdam --batch_size 40 --pos_encoding learnable --d_model 128 --seed 21
```
- Here, we performed pretraining for 300 epochs with a 7:3 ratio between the Trainset and Validationset.
- Additionally, the learning rate, optimizer (Adam, RAdam, AdamW), batch size, d_model, and other parameters are **customizable by the user**.
- After pretraining is complete, the configuration used for training, indexes of the data, metric values, and other relevant information are saved in the **`Experiments` folder**.
- You will find a `checkpoints` folder where the model with the best performance (key_metric: loss) during pretraining and the model from the last epoch are saved. --> **We will load the model with the best performance and proceed with FINE-TUNING**.

### 3. Fine-tuning for classification task
```bash
python src/main.py --output_dir finetuned --comment "finetune for classification" --name Sleep_finetuned --data_dir ./SleepData/ --data_class sleep --pattern TRAIN --val_ratio 0.2 --test_ratio 0.2 --epochs 200 --lr 0.0001 --optimizer RAdam --batch_size 40 --pos_encoding learnable --d_model 128 --load_model experiments/{folder_name_with_timepoint}/checkpoints/model_best.pth --task classification --change_output --key_metric precision --seed 21
```
- In the provided code, you can see the `--load_model` parameter contains `{folder_name_with_timepoint}`, which allows you to utilize various pretraining results for finetuning. Therefore, if you want to finetune the most recent pretraining model, you should refer to the folder names and select the one with the latest timestamp.
- Here, we performed training for the classification task over 200 epochs, using a 6:2:2 split for the Trainset, Validationset, and Testset. We specified precision as the key metric to select the best model based on precision, although it is **possible to configure it to use loss or accuracy instead**. Similar to pretraining, other configurations can also be customized by the user.
- One **important thing to note** is that **the model's structure must be maintained when loading the model**. For example, if the pretraining was done with d_model=64, it should not be changed to 128 during fine-tuning. The positional encoding must also be preserved.
- You will find a `checkpoints` folder where the model with the best performance (key_metric: precision) during pretraining and the model from the last epoch are saved. --> **We will load the model with the best performance and proceed with TEST**.

### 4. Test Model
```bash
python src/main.py --output_dir test_result --comment "test for classification" --name Sleep_test --data_dir ./SleepData/ --data_class sleep --pattern TEST --val_ratio 0.2 --test_ratio 0.2 --batch_size 40 --d_model 128 --load_model finetuned/{folder_name_with_timepoint}/checkpoints/model_best.pth --task classification --key_metric precision --test_only testset --freeze --pos_encoding learnable --seed 21
```
- For testing, you can load the model in a similar way as fine-tuning by modifying the `{folder_name_with_timepoint}` part of the `--load_model` parameter to point to the **`finetuned` folder**.
- **Important Notes**
  1. Just like in the fine-tuning part, you must **maintain the d_model and positional encoding**.
  2. The **train:validation:test ratio** and **seed value** from the fine-tuning part must be maintained. If these values change, the dataset composition will differ.
  3. The `--test_only testset` `--freeze` configuration must be included. This ensures that no training occurs and predictions are made using fixed parameters for performance evaluation.
  4. Set `--pattern` to TEST.

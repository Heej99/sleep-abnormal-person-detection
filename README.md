# sleep-abnormal-person-unsupervised-detection
수면의 특징을 반영한 representation learning으로 수면 이상 환자를 detect하는 Task를 수행했습니다. 

# 터미널에서 실행 방법
python src/main.py --output_dir experiments --comment "Training_Sleep_Data" --name Training_Sleep_Data --records_file Training_Sleep_records.xls --data_dir {OurData 파일 저장 폴더}/OurData/ --data_class sleep --pattern TRAIN --val_ratio 0.2 --epochs {total_epoch:숫자} --lr 0.001 --optimizer RAdam --batch_size 32 --pos_encoding learnable --d_model 64

- 추가  self.parser.add_argument('--mask_mode', choices={'separate', 'concurrent'}, default='separate',
                    help=("Imputation: whether each variable should be masked separately "
- mask mode : separate - 각 피처가 독립적으로 마스킹, concurrent - 같은 timestamp의 피처 같이 마스킹
- 여기서 concurrent 사용해야할듯?

# 텐서보드 사용 방법
1) 터미널에 tensorboard —-logdir={tb_summaries 폴더 경로} - experimet 안의 record 폴더 안에 있음
2) localhost:6006 - url입력
3) 실행예시
   
   <img width="740" alt="image" src="https://github.com/Heej99/sleep-abnormal-person-unsupervised-detection/assets/42797013/9e56d119-748d-458d-8d57-dffe35051f70">


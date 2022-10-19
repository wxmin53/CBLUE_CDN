import os
import sys
sys.path.append('baselines')
import argparse
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, AlbertModel, BertForSequenceClassification, \
    AlbertForSequenceClassification

from cblue import config
from cblue.models import CDNForCLSModel
from cblue.trainer import CDNForCLSTrainer, CDNForNUMTrainer
from cblue.utils import init_logger, seed_everything
from cblue.data import CDNDataset, CDNDataProcessor
from cblue.models import save_zen_model, ZenModel, ZenForSequenceClassification, ZenNgramDict


MODEL_CLASS = {
    'bert': (BertTokenizer, BertModel),
    'roberta': (BertTokenizer, BertModel),
    'albert': (BertTokenizer, AlbertModel),
    'zen': (BertTokenizer, ZenModel)
}

CLS_MODEL_CLASS = {
    'bert': BertForSequenceClassification,
    'roberta': BertForSequenceClassification,
    'albert': AlbertForSequenceClassification,
    'zen': ZenForSequenceClassification
}


def main():  
    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)
    output_dir = config.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.join(output_dir, config.model_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    logger = init_logger(os.path.join(output_dir, f'{config.task_name}_{config.model_name}.log'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device
    seed_everything(config.seed)

    if 'albert' in config.model_name:
        config.model_type = 'albert'

    # 'bert': (BertTokenizer, BertModel),
    tokenizer_class, model_class = MODEL_CLASS[config.model_type]

    if config.do_train:
        logger.info('Training CLS model...')
        tokenizer = tokenizer_class.from_pretrained(os.path.join(config.model_dir, config.model_name))

        ngram_dict = None
        if config.model_type == 'zen':
            ngram_dict = ZenNgramDict(os.path.join(config.model_dir, config.model_name), tokenizer=tokenizer)

        data_processor = CDNDataProcessor(root=config.data_dir, recall_k=config.recall_k,
                                          negative_sample=config.num_neg)
        train_samples, recall_orig_train_samples, recall_orig_train_samples_scores = data_processor.get_train_sample(dtype='cls', do_augment=config.do_aug)
        eval_samples, recall_orig_eval_samples, recall_orig_train_samples_scores = data_processor.get_dev_sample(dtype='cls', do_augment=config.do_aug)
        if data_processor.recall:
            logger.info('first recall score: %s', data_processor.recall)

        train_dataset = CDNDataset(train_samples, data_processor, dtype='cls', mode='train')
        eval_dataset = CDNDataset(eval_samples, data_processor, dtype='cls', mode='eval')

        # model与cls_model_class是什么区别
        model = CDNForCLSModel(model_class, encoder_path=os.path.join(config.model_dir, config.model_name),
                               num_labels=data_processor.num_labels_cls)
        cls_model_class = CLS_MODEL_CLASS[config.model_type]
        trainer = CDNForCLSTrainer(args=config, model=model, data_processor=data_processor,
                                   tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset,
                                   logger=logger, recall_orig_eval_samples=recall_orig_eval_samples,
                                   model_class=cls_model_class, recall_orig_eval_samples_scores=recall_orig_train_samples_scores,
                                   ngram_dict=ngram_dict)

        global_step, best_step = trainer.train()

        # 保存cls模型：取finetune效果最好的模型
        model = CDNForCLSModel(model_class, encoder_path=os.path.join(output_dir, f'checkpoint-{best_step}'),
                               num_labels=data_processor.num_labels_cls)
        model.load_state_dict(torch.load(os.path.join(output_dir, f'checkpoint-{best_step}', 'pytorch_model.pt')))
        tokenizer = tokenizer_class.from_pretrained(os.path.join(output_dir, f'checkpoint-{best_step}'))
        torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model_cls.pt'))
        if not os.path.exists(os.path.join(output_dir, 'cls')):
            os.mkdir(os.path.join(output_dir, 'cls'))

        if config.model_type == 'zen':
            save_zen_model(os.path.join(output_dir, 'cls'), model.encoder, tokenizer, ngram_dict, config)
        else:
            model.encoder.save_pretrained(os.path.join(output_dir, 'cls'))

        tokenizer.save_vocabulary(save_directory=os.path.join(output_dir, 'cls'))
        logger.info('Saving models checkpoint to %s', os.path.join(output_dir, 'cls'))

        logger.info('Training NUM model...')
        config.logging_steps = 30
        config.save_steps = 30
        train_samples = data_processor.get_train_sample(dtype='num', do_augment=1)
        eval_samples = data_processor.get_dev_sample(dtype='num')
        train_dataset = CDNDataset(train_samples, data_processor, dtype='num', mode='train')
        eval_dataset = CDNDataset(eval_samples, data_processor, dtype='num', mode='eval')

        cls_model_class = CLS_MODEL_CLASS[config.model_type]
        model = cls_model_class.from_pretrained(os.path.join(config.model_dir, config.model_name),
                                                num_labels=data_processor.num_labels_num)
        trainer = CDNForNUMTrainer(args=config, model=model, data_processor=data_processor,
                                   tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset,
                                   logger=logger, model_class=cls_model_class, ngram_dict=ngram_dict)

        global_step, best_step = trainer.train()

    if config.do_predict:
        tokenizer = tokenizer_class.from_pretrained(os.path.join(output_dir, 'cls'))

        ngram_dict = None
        if config.model_type == 'zen':
            ngram_dict = ZenNgramDict(os.path.join(config.model_dir, config.model_name), tokenizer=tokenizer)

        data_processor = CDNDataProcessor(root=config.data_dir, recall_k=config.recall_k,
                                          negative_sample=config.num_neg)
        # recall_orig_test_samples = {'text':[text1, text2, ...],
        #                             'recall_label': [[[],[],...], [[],[],...]],
        #                             'label': [[0],...]
        # test_samples = {'text1':[text1, text1, ..., text2, ...],
        #                 'text2':[label11, label12, ..., label21, ...],
        #                 'label':[0, 0, ..., 0, ...]}
        # recall_orig_test_samples_scores = [[t0k0, t0k1, t0kn], ..., [tmk0, tmk1, tmkn]]
        test_samples, recall_orig_test_samples, recall_orig_test_samples_scores = data_processor.get_test_sample(dtype='cls')

        test_dataset = CDNDataset(test_samples, data_processor, dtype='cls', mode='test')
        cls_model_class = CLS_MODEL_CLASS[config.model_type]

        model = CDNForCLSModel(model_class, encoder_path=os.path.join(output_dir, 'cls'),
                               num_labels=data_processor.num_labels_cls)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model_cls.pt')))
        trainer = CDNForCLSTrainer(args=config, model=model, data_processor=data_processor,
                                   tokenizer=tokenizer, logger=logger,
                                   recall_orig_eval_samples=recall_orig_test_samples,
                                   model_class=cls_model_class, ngram_dict=ngram_dict)
        cls_preds = trainer.predict(test_dataset, model)

        # cls_preds = np.load(os.path.join(result_output_dir, 'cdn_test_preds.npy'))

        # test_samples = ['text1'] = ["", "", ...]
        test_samples = data_processor.get_test_sample(dtype='num')
        orig_texts = data_processor.get_test_orig_text()
        test_dataset = CDNDataset(test_samples, data_processor, dtype='num', mode='test')
        model = cls_model_class.from_pretrained(os.path.join(output_dir, 'num'),
                                                num_labels=data_processor.num_labels_num)
        trainer = CDNForNUMTrainer(args=config, model=model, data_processor=data_processor,
                                   tokenizer=tokenizer, logger=logger,
                                   model_class=cls_model_class, ngram_dict=ngram_dict)
        trainer.predict(model, test_dataset, orig_texts, cls_preds, recall_orig_test_samples,
                        recall_orig_test_samples_scores)


if __name__ == '__main__':
    main()

'''
Dataset Module to load custom datasets
'''

from operator import index
from textwrap import indent
#import numpy as np
import os
import pandas as pd


class Dataset:
    '''
    This module is designed to help users load their own custom dataset.
    '''
    def __init__(self, path:str, encoding:str='utf-8') -> None:
        '''
        initialization of Dataset
        :param path : string, path to the dataset
        :param encoding : string, encoding to read data (default 'utf-8')
        '''

        self.initialize_corpus(
            self.load_data(path, encoding)) # initialize train, test, dev
        self.load_vocab(path, encoding) # get vocabulary
        self.wordtoindex = {word: index for index, word in enumerate(self.vocab)}
        self.indextoword = {index: word for word, index in self.wordtoindex.items()}
        self.count_words()

    def initialize_corpus(self, data:dict) -> None:
        self.train_corpus = data['train_corpus']
        self.test_corpus = data['test_corpus']
        self.dev_corpus = data['dev_corpus']

        self.train_labels = data['train_labels']
        self.test_labels = data['test_labels']
        self.dev_labels = data['dev_labels']

    def load_data(self, path:str, encoding:str) -> dict:
        data = {
            'train_corpus': [],
            'test_corpus': [],
            'dev_corpus': [],
            'train_labels': [],
            'test_labels': [],
            'dev_labels': []
        }
        
        try:
            df = pd.read_csv(f'{path}/data.tsv', delimiter='\t', encoding=encoding)
            
            train = df[df['split'] == 'train']
            test = df[df['split'] == 'test']
            dev = df[df['split'] == 'dev']
            
            data['train_corpus'].extend(train['text'])
            data['test_corpus'].extend(test['text'])
            data['dev_corpus'].extend(dev['text'])
            
            data['train_labels'].extend(train['label'])
            data['test_labels'].extend(test['label'])
            data['dev_labels'].extend(dev['label'])
            
            return data
        
        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")


    def load_vocab(self, path:str, encoding:str) -> None:
        self.vocab = ['UNK']
        file_path = f'{path}/vocab.txt'
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
                for line in lines:
                    _ = line.split()
                    self.vocab.append(_[0])
        else:
            try:
                df = pd.read_csv(f'{path}/data.tsv', delimiter='\t', encoding=encoding)
                text_column_name = 'text'

                # Check if the specified text column exists in the DataFrame
                if text_column_name not in df.columns:
                    raise ValueError(f"'{text_column_name}' column not found in the DataFrame.")
                    
                text_column = df[text_column_name]
                
                vocab_set = set()

                for text in text_column:
                    if pd.notna(text):  # Check for non-null values
                        words = text.split()
                        vocab_set.update(words)

                vocab_list = sorted(list(vocab_set))
                self.vocab = vocab_list
            
            except FileNotFoundError as e:
                print(f"File not found: {e}")
            except Exception as e:
                print(f"An error occurred: {str(e)}")
    
    def count_words(self):
        self.words_count = {}
        
        for doc in self.train_corpus:
            tokenized = doc.split()
            for token in tokenized:
                if token in self.vocab:
                    try:
                        self.words_count[token] += 1
                    except:
                        self.words_count[token] = 1

        for i in list(self.words_count.keys()):
            if self.words_count[i] == 0:
                del self.words_count[i]
                del self.indextoword[self.wordtoindex[i]]
                del self.wordtoindex[i]

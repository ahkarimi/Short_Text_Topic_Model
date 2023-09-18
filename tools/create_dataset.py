"""
    This is the function of the scraper that generates a dataset from a list of hashtags.
"""

from scraper import TwitterScraper
from hazm import word_tokenize, Normalizer, Lemmatizer
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import json

normalizer = Normalizer().normalize
lemmatizer = Lemmatizer().lemmatize

# Retrieved from https://github.com/kharazi/persian-stopwords
stopwords = set(open('stop_words/stop_words.txt', encoding='utf8').read().splitlines())
# Retrieved from https://github.com/amirshnll/Persian-Swear-Words
swearing_words = set(open('stop_words/swear_words.txt', encoding='utf8').read().splitlines())

bad_hashtags = set(['تا_آخوند_کفن_نشود_این_وطن_وطن_نشود',
'ایران_را_پس_میگیریم',
'جمهوری_اسلامی_نابود_باید_گردد',
'مرگ_بر_خامنه\\u200cای_جنایتکار',
'مرگ_بر_کلیت_و_تمامیت_جمهوری_اسلامی',
'جاویدشاه',
'نه_به_جمهورى_اسلامى',
'ریدم_تو_اسلام',
'براندازم',
'قيام_تا_سرنگونی',
'مريم_رجوی'])

swearing_words.update(bad_hashtags)

class const:
    farsi = ('ب', 'س', 'ش', 'ل', 'ت', 'ن', 'م', 'گ', 'ظ', 'ط', 'ز',
             'ر', 'ژ', 'ذ', 'د', 'پ', 'چ', 'ج', 'ح', 'ع', 
             'خ', 'غ', 'ف', 'ق', 'ث', 'ص', 'ض','\u0020',
             '\u200C', '\u060c','؟', '!', '?', '.', ':','\n', '_')

    alef = ('ا', 'آ', 'ء', 'أ', 'إ')
    vav = ('و', 'ؤ')
    heh = ('ه', 'ة', 'ە')
    yah = ('ی', 'ي', 'ئ', 'ى')
    kaf = ('ک', 'ك')


def remover(char):
    if char in const.farsi:
        return char
    if char in const.alef:
        return const.alef[0]
    if char in const.vav:
        return const.vav[0]
    if char in const.heh:
        return const.heh[0]
    if char in const.yah:
        return const.yah[0]
    if char in const.kaf:
        return const.kaf[0]
    return ''


def pre_process(text):
    persian_words = map(remover, text)
    sentence = ''.join(persian_words)
    if (len(sentence) < 20):
      return None
    word_tokens = word_tokenize(sentence)

    for w in word_tokens:
      if w in swearing_words:
        return None

    filtered_stopwords = [w for w in word_tokens if w not in stopwords and len(w) > 1]

    if (len(filtered_stopwords) < 5):
      return None
    filtered_stopwords = ' '.join(filtered_stopwords)
    return filtered_stopwords


def main(args):
    df = pd.DataFrame([])
    with open(args.hashtags, "r", encoding="utf-8") as json_file:
        hashtags = json.load(json_file)

    for topic in tqdm(hashtags.keys(), desc='Scraping Topics'):
        scraper = TwitterScraper(
            max_results=args.max_results,
            hashtags=hashtags[topic],
            lang=args.lang,
            until=args.until,
            since=args.since,
            with_replies=args.with_replies,
        )
        result = scraper.basic_mode()
        result['topic'] = topic
        df = pd.concat([df, result], axis=0)
    
    # preprocess
    df = df[df['username'].notna()]
    tweets = map(pre_process, df.text)
    tweets = list(tweets)
    df['processed_text'] = tweets
    df = df[df['processed_text'].notna()]
    df = df.reset_index(drop=True)

    df = df.drop_duplicates(subset='tweet_id')
    print('-- Dataframe shape: {}'.format(df.shape))
    df = df.groupby('topic').apply(lambda x: x.sample( len(x) if len(x) < 10000 else 10000)).reset_index(drop=True)
    df = df.reset_index(drop=True)

    df.to_csv("datasets/twitter_dataset.tsv", index=False, sep='\t')
    print('[ OK ] Dataset created.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-results", default=(2 * (10 ** 4)), type=int)
    parser.add_argument("--lang", default="fa", type=str)
    parser.add_argument("--until", default="2022-02-10", type=str)
    parser.add_argument("--since", default="2019-06-01", type=str)
    parser.add_argument("--with-replies", default=False, type=bool)
    parser.add_argument("--hashtags", default='stop_words/local_hashtags.json')
    args = parser.parse_args()
    print(args)
    main(args)

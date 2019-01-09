# -*- coding: utf-8 -*-
#--------- NLP Processing, need Python 3  and Java JPype  ----------------------------
import IPython, datetime, os, sys
import numpy as np, pandas as pd, scipy as sci
import shutil, urllib3
from bs4 import BeautifulSoup





################################################################################
#-------- Japanese Utilitie--------------------------------------------------------
import re
# Regular expression unicode blocks collected from
# http://www.localizingjapan.com/blog/2012/01/20/regular-expressions-for-japanese-text/
hiragana_full = r'[ぁ-ゟ]'
katakana_full = r'[゠-ヿ]'
kanji = r'[㐀-䶵一-鿋豈-頻]'
radicals = r'[⺀-⿕]'
katakana_half_width = r'[｟-ﾟ]'
alphanum_full = r'[！-～]'
symbols_punct = r'[、-〿]'
misc_symbols = r'[ㇰ-ㇿ㈠-㉃㊀-㋾㌀-㍿]'
ascii_char = r'[ -~]'
'''
hiragana_full = ur'[\\u3041-\\u3096]'
katakana_full = ur'[\\u30A0-\\u30FF]'
kanji = ur'[\\u3400-\\u4DB5\\u4E00-\\u9FCB\\uF900-\\uFA6A]'
radicals = ur'[\\u2E80-\\u2FD5]'
half_width = ur'[\\uFF5F-\\uFF9F]'
alphanum_full = ur'[\\uFF01-\\uFF5E]'
symbols_punct = ur'[\x3000-\x303F]'
misc_symbols = ur'[\x31F0-\x31FF\x3220-\x3243\x3280-\x337F]'
'''

def ja_extract_unicode_block(unicode_block, string):
    ''' extracts and returns all texts from a unicode block from string argument. '''
    return re.findall( unicode_block, string)

def ja_remove_unicode_block(unicode_block, string):
    ''' removes all cha from a unicode block and return remaining texts from string  '''
    return re.sub( unicode_block, ' ', string)

def ja_getkanji(vv):
 vv= remove_unicode_block(hiragana_full, vv)
 vv= remove_unicode_block(katakana_full, vv)
 vv= remove_unicode_block(radicals, vv)
 vv= remove_unicode_block(katakana_half_width, vv)
 vv= remove_unicode_block(symbols_punct, vv)
 vv= remove_unicode_block(misc_symbols, vv)
 vv= remove_unicode_block(alphanum_full, vv)
 ff= vv.split(' '); vv=''
 for aa in ff:
    if not aa=='': vv+= ' '+ aa
 return vv

# text = '初めての駅 自、りると、ママは、トットちゃん㈠㉃㊀㋾㌀㍿'
#  remove_unicode_block(kanji, text))
# ''.join(extract_unicode_block(hiragana_full, text)))


# Mode 1 : Get all the prununciation sentence
def ja_getpronunciation_txten(txt):
 import java, romkan
 ll= java.japanese_tokenizer_kuromoji(txt,  parsermode="NORMAL")
 vv= ''
 for tt in ll:
   if tt[8] !='' :
    vv= (vv + ' '+ (romkan.to_roma(tt[8]))).strip()
   if tt[0]=='。' : vv= vv + '\n\n'
 return vv



# Mode 2 : Get all the prununciation for each Kanji
def ja_getpronunciation_kanji(txt, parsermode="SEARCH"):
 import java, romkan
 txt= remove_unicode_block(symbols_punct, txt)
 txt= remove_unicode_block(misc_symbols, txt)
 txt= remove_unicode_block(alphanum_full, txt)
 ll2= java.japanese_tokenizer_kuromoji(txt,  parsermode=parsermode)
 vv= ''
 for tt in ll2:
  if "".join(extract_unicode_block(kanji, tt[0])) != '' :
   if (tt[7] !=''   and ( tt[1]=='動詞'  or tt[1]=='名詞' )):
    vv= vv + ' '+ tt[0] +  ' '+ romkan.to_roma(tt[8])  + '\n'
 return vv


def ja_importKanjiDict(file2):
 import csv
 with open(file2, 'r',encoding='UTF-8',) as f:
  reader = csv.reader(f,  delimiter='/' )
  kanjidict=  dict(reader)
 return kanjidict

#kanjidict= importKanjiDict(r"E:/_data/japanese/edictnewF.txt")
#kanjidict['拳固']

kanjidictfile= r"E:/_data/japanese/edictnewF.py"

# Mode 2 : Get all the prununciation for each Kanji
def ja_getpronunciation_kanji3(txt, parsermode="SEARCH"):
 import java, romkan
 txt= remove_unicode_block(symbols_punct, txt)
 txt= remove_unicode_block(misc_symbols, txt)
 txt= remove_unicode_block(alphanum_full, txt)
 ll2= java.japanese_tokenizer_kuromoji(txt,  parsermode=parsermode)
 vv= ''
 kanjidict= importKanjiDict(kanjidictfile)
 for tt in ll2:
  if "".join(extract_unicode_block(kanji, tt[0])) != '' :
   if (tt[7] !=''  ):
    name= tt[0]
    try :
     vv= vv + ' '+ name +' : '+ romkan.to_roma(tt[8])+ " : " + kanjidict[name]+'\n'
    except:
     pass
 return vv


# Mode 3 : Get all the prununciation sentence
def ja_gettranslation_textenja(txt):
 import java
 ll= java.japanese_tokenizer_kuromoji(txt,  parsermode="SEARCH")
 kanjidict= importKanjiDict(kanjidictfile)
 vv= ''; vv2=''; xx=''
 for tt in ll:
   name= tt[0]
   if tt[8] !='' :
    vv2= (vv2 + '  '+ (name)).strip()
    if (tt[7] !=''   and ( tt[1]=='動詞'  or tt[1]=='名詞' )):
     try:
      vv= (vv + ' '+ (kanjidict[name])).strip() + ' /'
     except KeyError:
      pass
   if name=='。' :
       xx= xx + vv2 + '\n' + vv + '\n\n'
       vv= ''; vv2=''
 return xx






# Mode 3 : Get all the prununciation sentence
def ja_getpronunciation_textenja(txt):
 import java, romkan
 ll= java.japanese_tokenizer_kuromoji(txt,  parsermode="NORMAL")
 vv= ''; vv2=''; xx=''
 for tt in ll:
   if tt[8] !='' :
    vv2= (vv2 + '  '+ (tt[0])).strip()
    vv= (vv + ' '+ (romkan.to_roma(tt[8]))).strip()
   if tt[0]=='。' :
       xx= xx + vv2 + '\n' + vv + '\n\n'
       vv= ''; vv2=''
 return xx


# Send Text Pronunciation by email
def ja_send_textpronunciation(url1, email1):
 aa= web_gettext_fromurl(url1)
 kk= ja_getpronunciation_kanji3(aa)
 mm= ja_getpronunciation_textenja(aa)
 mm2= ja_gettranslation_textenja(aa)
 mm= mm + '\n\n\n' + kk + '\n\n\n' + mm2
 send_email("Kevin", email1, "JapaneseText:"+mm[0:20] ,mm  )


# Send Text Pronunciation by email
def ja_sendjp(url1): ja_send_textpronunciation(url1)



def ja_weblio():
 '''
 #!/usr/bin/env python
# -*- coding: utf-8 -*-

from requests import get, RequestException
from bs4 import BeautifulSoup


class Weblio:

    """
        Weblio API(as if) consumer
    """

    def __init__(self):
        #self.definition_url = 'http://www.weblio.jp/content/%s'
        # Cannot find some rare words
        self.definition_url = 'http://ejje.weblio.jp/english-thesaurus/%s'
        self.lookup_url = 'http://ejje.weblio.jp/content/%s'
        self.examples_url = 'http://ejje.weblio.jp/sentence/content/%s'
        # TODO: stats (and corresponding DB entity)
        self.stats = {}

    def definition(self, term):
        """Fetches definitions and similar words, synonyms"""
        data = self.process(self.definition_url, term)
        # TODO: implement!
        if data:
            # Check for different possible divs:
            # NetDicBody
            # Wiktionary
            gloss = data.find('div', 'NetDicBody')
            #print gloss.getText()
            definitions = gloss.getText().split('(')
            print definitions

    def lookup(self, term):
        """Fetches translations (jp-en) for different use-cases"""
        pass

    def examples(self, term, number=4, portion=4, tuples=False):
        """
        Fetches examples from Weblio
        :param term:    word or phrase to lookup
        :param number:  number of examples to fetch
        :param portion: portion of examples to use (e.g., 1/2 -> from the middle)
        :returns:       list of touples (example, translation)
        """
        data = self.process(self.examples_url, term)
        examples = [] if tuples else {}
        if data:
            #for example in data.find_all('div', 'qotC')[-number:]:
            total = data.find_all('div', 'qotC')
            n = len(total) / portion
            # Let's take examples from the middle (TODO: golden ratio?)
            for example in total[n: n + number]:
                # TODO: remove identical examples or similar to term
                # TODO: if no examples found -> log it (and mark term)
                # TODO: check (term:example) when there's english example [0] instead
                sentence = example.contents[0].getText()
                source = example.contents[1].span.getText()
                translation = example.contents[1].getText().replace(source, '')
                translation = self.remove_comments(translation, '<!--')
                if tuples:
                    examples.append((sentence, translation))
                else:
                    #examples.append(
                        #{'example': sentence, 'translation': translation}
                    #)
                    examples[sentence] = translation

        return examples

    def process(self, url, term):
        try:
            return BeautifulSoup(get(url % term).text, 'lxml')
        except RequestException:
            return None

    def remove_comments(self, line, sep):
        """Trims comments from string"""
        for s in sep:
            line = line.split(s)[0]
        return line.strip()

 '''






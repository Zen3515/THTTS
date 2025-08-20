# src: https://github.com/VYNCX/F5-TTS-THAI/blob/99b8314f66a14fc2f0a6b53e5122829fbdf9c59c/src/f5_tts/cleantext/TH2IPA.py

import re
import html

from tltk import g2p as tltkg2p
from pythainlp import word_tokenize
from langdetect import detect
from phonemizer import phonemize
from pythainlp.util import expand_maiyamok

from .raw_dict import NUMBER2PHONE_DICT, THAI2PHONE_DICT, PHONE2IPA, PHONE2HAAS, PHONE2RTGS, PHONE2RTGS_CODA, HAAS2PHONE


def eng_ipa(text):
    ipa = phonemize(
        text,
        language='en-us',
        backend='espeak',
        strip=True,
        punctuation_marks=';:,.!?¡¿—…"«»“”()',
        preserve_punctuation=True,
        with_stress=True
    )
    return ipa


def ENG2IPA(text):
    ipa_result = eng_ipa(text)
    return ipa_result


def clean_text(text):
    return re.sub(r'[^\u0E00-\u0E7F\s]', '', text).strip()


def th_to_g2p(text):
    cleaned_text = clean_text(text)
    cleaned_text = expand_maiyamok(cleaned_text)  # Expand Maiyamok characters
    result = g2p(cleaned_text, 'ipa')
    return result


def any_ipa(text):
    lang = detect(text)
    if lang == "th":
        ipa_text = th_to_g2p(text)
    elif lang == "en":
        ipa_text = ENG2IPA(text)
    else:
        ipa_text = ENG2IPA(text)

    return ipa_text


##################################################
# G2P FUNCTIONS
##################################################

SHORT_VOWELS = "aivueyoxz"
LONG_VOWELS = "AIVUEYOXZ"
DIPHTHONGS = "JWR"
VOWELS = SHORT_VOWELS + LONG_VOWELS + DIPHTHONGS
CLUSTERS = ["br", "bl", "pr", "pl", "Pr", "Pl", "fr", "fl", "dr", "tr", "Tr", "kr", "kl", "kw", "Kr", "Kl", "Kw"]
ONSETS = ["b", "p", "P", "m", "f", "d", "t", "T", "n", "s", "r", "l", "c", "C", "k", "K", "N", "w", "j", "h", "?"]
CODAS = ["p", "m", "f", "t", "d", "n", "s", "l", "c", "k", "N", "w", "j", "?", "-"]


def validate(phone: str):
    """validate encoded phone
    >>> validate('paj1 dAj3') -> True
    >>> validate('aaa aaa') -> False
    """
    syls = phone.split()
    for syl in syls:
        try:
            tone = syl[-1]  # prA-1 -> 1
            coda = syl[-2]  # prA-1 -> -
            vowel = syl[-3]  # prA-1 -> A
            onset = syl[:-3]  # prA-1 -> pr
        except:
            return False
        # check all 4 parts are valid
        if tone in '12345' and coda in CODAS and vowel in VOWELS and onset in CLUSTERS+ONSETS:
            continue
        else:
            return False
    return True


def decode(phone, transcription='haas', keep_space: bool | str = True):
    """decode phone into Haas or IPA

    Parameters
    ----------
        phone : str or list
            encoded syllbles
            e.g. 'kot2 mAj5' or ['kot2', 'mAj5']
        transcription : str
            'haas'(default) or 'ipa' or 'RTGS'

    Return
    ------
    str
        decoded phone, e.g. kòt mǎːj

    Example
    -------
        decode('kot2 mAj5 ʔA-1 jA-1 123', 'ipa')
            'kòt mǎːj ʔaː jaː 123'
    """

    # check type of parameter
    if type(phone) == str:  # one string e.g. "kot2 mAj5"
        syls = phone.split()
    elif type(phone) == list:  # ['kot2', 'mAj5']
        syls = phone
    else:
        raise TypeError

    decoded_syls = []
    for i, syl in enumerate(syls):
        if not validate(syl):  # e.g. English, punctuation
            decoded_syls.append(syl)  # return original string
            continue
        tone = syl[-1]
        coda = syl[-2]
        coda = coda.replace('ʔ', '-')  # delete ? in all codas
        """ # delete ? unless the final syllable
        if i != len(syls) - 1: 
            coda = coda.replace('ʔ','-') # "-" = no coda
        """
        vowel = syl[-3]
        onset = syl[:-3]  # one or two characters
        if transcription.lower() == 'ipa':
            decoded_syls.append(''.join([PHONE2IPA[c] for c in onset]) + PHONE2IPA[vowel+tone] + PHONE2IPA[coda])
        elif transcription.lower() == 'haas':
            decoded_syls.append(''.join([PHONE2HAAS[c] for c in onset]) + PHONE2HAAS[vowel+tone] + PHONE2HAAS[coda])
        elif transcription.lower() == 'rtgs':
            decoded_syls.append(''.join([PHONE2RTGS[c] for c in onset]) + PHONE2RTGS[vowel+tone] + PHONE2RTGS_CODA[coda])
    if keep_space == False:
        decoded_syls = ''.join(decoded_syls)
    elif type(keep_space) == str:  # custom delimiter
        decoded_syls = keep_space.join(decoded_syls)
    else:
        decoded_syls = ''.join(decoded_syls)
    return decoded_syls

# tokenize by pythainlp -> look up dictionary
# if there is none, try to use tltk instead


def g2p(sentence, transcription='haas', return_tokens=False, decoded=True):
    """G2P function for Thai sentence

    Parameters
    ----------
    sentence : str or list
        string of Thai sentences or list of tokenized words 
    transcription : str
        'haas'(default) or 'ipa' or 'rtgs'
    return_tokens : bool
        whether returns also tokenized sentence
    decoded : bool
        if True, returns decoded phone e.g. paj roːŋ rian
        if False, returns undecoded phone e.g. paj1 rON1 rJn1

    Return
    ------
    str
        syllables delimited by whitespaces 
    or list
        list of [token, phone]

    Examples
    --------
        g2p('ไปโรงเรียน')
            'paj rooŋ rian'

        g2p('ไปโรงเรียน', transcription='ipa')
            'paj roːŋ rian'

        g2p('ไปโรงเรียน', return_tokens=True)
            [['ไป', 'pay'], ['โรงเรียน', 'rooŋ rian']]
    """

    ### tokenize ###
    if type(sentence) == str:  # input is string
        sentence = clean(sentence)  # preprocessing
        tokens = word_tokenize(sentence, keep_whitespace=False)
    elif type(sentence) == list and type(sentence[0]) == str:  # input is tokens already
        tokens = sentence

    token_phone_list: list[list[str]] = []  # list of [token, phone] e.g. [['ไป','paj1'],['โรงเรียน','rON1 rJn1']]

    ### check each token ###
    for i, token in enumerate(tokens):

        # exceptions

        if token == 'น.' and i > 0 and\
                (token_phone_list[-1][1].endswith('nA-1 li-4 kA-1') or token_phone_list[-1][1].endswith('nA-1 TI-1')):
            token_phone_list[-1][0] += ' น.'  # add to previous token to avoid duplicate
            continue
        elif token == 'ๆ' and i > 0:  # if single ๆ, repeat final one
            token_phone_list[-1][0] += ' ๆ'
            token_phone_list[-1][1] += ' ' + token_phone_list[-1][1]
            continue

        # Thai word in dictionary
        elif token in THAI2PHONE_DICT:
            phone: str = get_phone_word(token)  # pyright: ignore[reportAssignmentType]

        # single thai character (maybe mistake of tokenization) -> pass
        elif re.match('[ก-ฮ]$', token):
            continue

        # thaiword, but not in dictionary -> use tltk instead
        elif re.match(r'[ก-๙][ก-๙\-\.]*$', token):
            # phone = None  # return None, USE THIS LINE WHEN TEST
            phone = get_phone_word_tltk(token)

        # time e.g. 22.34
        elif is_time(token):
            phone = get_phone_time(token)

        # number
        elif is_number(token):
            phone = get_phone_number(token)

        # return original token, e.g. english, punctuation...
        else:
            phone = token

        token_phone_list.append([token, phone])

    ### decode ###
    if decoded:
        token_phone_list = [[t, decode(p, transcription)] for t, p in token_phone_list]

    ### return ###
    if return_tokens:
        return token_phone_list  # return as list of [token, phone]
    else:
        return ' '.join([phone for _, phone in token_phone_list])


def clean(text: str):
    text = html.unescape(text)
    text = re.sub(r'[\n\s]+', ' ', text)  # shrink whitespaces
    text = re.sub(r'https?://[^\s]+((?=\s)|(?=$))', '', text)  # remove URL
    text = re.sub(r'\((.+?)\)', r'( \1 )', text)  # add space before/after parentheses
    text = re.sub(r'\"(.+?)\"', r'" \1 "', text)  # add space before/after quotation
    text = re.sub(r'[“”„]', '"', text)  # convert double quotations into "
    text = re.sub(r'[‘’`]', "'", text)  # convert single quotations into '
    text = re.sub(r'[ \u00a0\xa0\u3000\u2002-\u200a\t]+', ' ', text)  # e.g. good  boy -> good boy
    text = re.sub(r'[\r\u200b\ufeff]+', '', text)  # remove non-breaking space
    return text.strip()


def get_phone_word(thaiword: str):
    # if the word in the dict, return the phone
    # ไป -> paj1
    return THAI2PHONE_DICT.get(thaiword, None)


def get_phone_word_tltk(thaiword: str):
    # if the word is not in dict, use tltk instead
    # tltk may return several sentences e.g. <tr/>paj0|maj4|<s/><tr/>maj2|paj0|<s/>
    # sentences = ['paj0|maj4', 'maj2|paj0']
    decoded_syls = []
    result = tltkg2p(thaiword)
    tokens = re.findall(r'<tr/>(\S+?)\|(?:<s/>|\s)', result)
    for token in tokens:  # 'paj0|maj4'
        # split to each syllable 'paj0', 'maj4'
        # delimiter : | or ^ or ~ '
        for syl in re.split(r"[|^~\']", token):
            syl = syl.replace('\\', '')  # remove \ e.g. เจิ้น -> c\\@n2
            ### change encoding ###
            tone = str(int(syl[-1])+1)  # 0->1 because use 0-4 in tltk
            if int(tone) > 5:  # strangely, there are tone "8" in tltk
                tone = str(int(tone)-5)
            syl = syl[:-1] + tone
            # replace vowels
            syl = re.sub(r'iia(?=\d)', 'J-', syl)  # /ia/
            syl = re.sub(r'iia', 'J', syl)
            syl = re.sub(r'ia', 'J-', syl)
            syl = re.sub(r'UUa(?=\d)', 'W-', syl)  # /ɯa/
            syl = re.sub(r'UUa', 'W', syl)
            syl = re.sub(r'Ua', 'W-', syl)
            syl = re.sub(r'uua(?=\d)', 'R-', syl)  # /ua/
            syl = re.sub(r'uua', 'R', syl)
            syl = re.sub(r'ua', 'R-', syl)
            syl = re.sub(r'aa(?=\d)', 'A-', syl)  # no coda
            syl = re.sub(r'aa', 'A', syl)  # with coda
            syl = re.sub(r'a(?=\d)', 'a-', syl)  # no coda
            syl = re.sub(r'ii(?=\d)', 'I-', syl)
            syl = re.sub(r'ii', 'I', syl)
            syl = re.sub(r'i(?=\d)', 'i-', syl)
            syl = re.sub(r'UU(?=\d)', 'V-', syl)  # /ɯ/
            syl = re.sub(r'UU', 'V', syl)
            syl = re.sub(r'U(?=\d)', 'v-', syl)
            syl = re.sub(r'U', 'v', syl)
            syl = re.sub(r'uu(?=\d)', 'U-', syl)  # /u/
            syl = re.sub(r'uu', 'U', syl)
            syl = re.sub(r'u(?=\d)', 'u-', syl)
            syl = re.sub(r'xx(?=\d)', 'Y-', syl)  # /ɛ/
            syl = re.sub(r'xx', 'Y', syl)
            syl = re.sub(r'x(?=\d)', 'y-', syl)
            syl = re.sub(r'x', 'y', syl)
            syl = re.sub(r'ee(?=\d)', 'E-', syl)  # /e/
            syl = re.sub(r'ee', 'E', syl)
            syl = re.sub(r'e(?=\d)', 'e-', syl)
            syl = re.sub(r'OO(?=\d)', 'X-', syl)  # /ɔ/
            syl = re.sub(r'OO', 'X', syl)
            syl = re.sub(r'O(?=\d)', 'x-', syl)
            syl = re.sub(r'O', 'x', syl)
            syl = re.sub(r'oo(?=\d)', 'O-', syl)  # /o/
            syl = re.sub(r'oo', 'O', syl)
            syl = re.sub(r'o(?=\d)', 'o-', syl)
            syl = re.sub(r'@@(?=\d)', 'Z-', syl)  # /ə/
            syl = re.sub(r'@@', 'Z', syl)
            syl = re.sub(r'@(?=\d)', 'z-', syl)
            syl = re.sub(r'@', 'z', syl)
            # replace consonants
            syl = re.sub(r'th', 'T', syl)
            syl = re.sub(r'kh', 'K', syl)
            syl = re.sub(r'ph', 'P', syl)
            syl = re.sub(r'ch', 'C', syl)

            decoded_syls.append(syl)

    return ' '.join(decoded_syls)


def is_time(text: str):
    # 8:00, 09.12, 12:12, 23.31น., etc
    return bool(re.match(r'[012]?[0-9][:\.][0-5][0-9](\s*)?(น.)?', text))


def get_phone_time(time: str):
    # 20.31 -> jI-3 sip2 nA-1 liʔ4 kA-1 sAm5 sip2 ʔet2 nA-1 TI-1
    hour, minute = re.split(r'[:\.]', time)  # 23.31น. -> [23, 31น.]
    minute = minute.split('น.')[0]  # 31น. -> 31
    if minute == '00':
        return get_phone_number(hour) + ' nA-1 li-4 kA-1'  # 8.00 -> pYt2 nA-1 li-4 kA-1
    else:
        return get_phone_number(hour) + ' nA-1 li-4 kA-1 ' + get_phone_number(minute) + ' nA-1 TI-1'


def is_number(text: str):
    return bool(re.match(r'\-?\d[\d\,]*(?:\.\d+)?$', text))


def get_phone_number(number: str):
    # 3,120 -> sAm5 Pan1 rXj4 jI-3 sip2
    # 123.123 -> nɯŋ2 rXj4 jI-3 sip2 sAm5 cut2 nɯŋ2 sXŋ5 sAm5
    number = str(number)  # float 123.5 -> str "123.5"
    if re.match(r'0[0-9]*[1-9]+', number):  # e.g. 0012 (exclude 0, 00)
        number = number.lstrip('0')  # 0012 -> 12
    number = number.replace(',', '')  # 1,000 -> 1000
    minus = number[0] == '-'  # bool to check negative
    number = number.strip('-')  # delete initial -
    if '.' not in number:  # if integer
        length = len(number)
        if length <= 2:
            if number in NUMBER2PHONE_DICT:
                phone = NUMBER2PHONE_DICT[number]
            else:
                phone = NUMBER2PHONE_DICT[number[0]+'0'] + ' ' + NUMBER2PHONE_DICT[number[1]]  # 34 -> 30 + 4
        elif length <= 7:  # 7 = million = ล้าน
            if number in NUMBER2PHONE_DICT:
                phone = NUMBER2PHONE_DICT[number]
            else:
                phone = NUMBER2PHONE_DICT[number[0]+'0'*(length-1)] + ' ' + get_phone_number(number[1:])  # 345 -> 300 + 45 (recursive)
        elif length <= 12:  # 12 = trillion = ล้านล้าน
            # 123456000 -> 123 + ล้าน + 456000
            upper = number[:-6]
            lower = number[-6:]  # xxx ล้าน
            if lower == '000000':
                phone = get_phone_number(upper) + ' lAn4'
            else:
                phone = get_phone_number(upper) + ' lAn4 ' + get_phone_number(lower)
        else:
            return number  # longer than 12, return original
    else:  # if decimal
        integer, decimal = number.split('.')
        decimal = ' '.join([get_phone_number(x) for x in decimal])  # get one by one
        phone = get_phone_number(integer) + ' cut2 ' + decimal
    if minus:
        return 'lop4 ' + phone
    else:
        return phone

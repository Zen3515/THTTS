# src: https://github.com/VYNCX/F5-TTS-THAI/blob/99b8314f66a14fc2f0a6b53e5122829fbdf9c59c/src/f5_tts/cleantext/th_repeat.py
# src: https://github.com/VYNCX/F5-TTS-THAI/blob/99b8314f66a14fc2f0a6b53e5122829fbdf9c59c/src/f5_tts/cleantext/number_tha.py
import re

from pythainlp.tokenize import syllable_tokenize


def remove_symbol(text: str):
    symbols = r"{}[]()-_?/\\|!*%$&@#^<>+-\";:~\`=“”"
    for symbol in symbols:
        text = text.replace(symbol, '')
    text = text.replace(" ๆ", "ๆ")
    return text


def process_thai_repeat(text: str):

    cleaned_symbols = remove_symbol(text)

    words = syllable_tokenize(cleaned_symbols)

    result = []
    i = 0
    while i < len(words):
        if i + 1 < len(words) and words[i + 1] == "ๆ":
            result.append(words[i])
            result.append(words[i])
            i += 2
        else:
            result.append(words[i])
            i += 1

    return "".join(result)


def number_to_thai_text(num, digit_by_digit=False):
    # Thai numerals and place values
    thai_digits = {
        0: "ศูนย์", 1: "หนึ่ง", 2: "สอง", 3: "สาม", 4: "สี่",
        5: "ห้า", 6: "หก", 7: "เจ็ด", 8: "แปด", 9: "เก้า"
    }
    thai_places = ["", "สิบ", "ร้อย", "พัน", "หมื่น", "แสน", "ล้าน"]

    # Handle zero case
    if num == 0:
        return thai_digits[0]

    # If digit_by_digit is True, read each digit separately
    if digit_by_digit:
        return " ".join(thai_digits[int(d)] for d in str(num))

    # For very large numbers, we'll process in chunks of millions
    if num >= 1000000:
        millions = num // 1000000
        remainder = num % 1000000
        result = number_to_thai_text(millions) + "ล้าน"
        if remainder > 0:
            result += number_to_thai_text(remainder)
        return result

    # Convert number to string and reverse it for easier place value processing
    num_str = str(num)
    digits = [int(d) for d in num_str]
    digits.reverse()  # Reverse to process from units to highest place

    result = []
    for i, digit in enumerate(digits):
        if digit == 0:
            continue  # Skip zeros

        # Special case for tens place
        if i == 1:
            if digit == 1:
                result.append(thai_places[i])  # "สิบ" for 10-19
            elif digit == 2:
                result.append("ยี่" + thai_places[i])  # "ยี่สิบ" for 20-29
            else:
                result.append(thai_digits[digit] + thai_places[i])
        # Special case for units place
        elif i == 0 and digit == 1:
            if len(digits) > 1 and digits[1] in [1, 2]:
                result.append("เอ็ด")  # "เอ็ด" for 11, 21
            else:
                result.append(thai_digits[digit])
        else:
            result.append(thai_digits[digit] + thai_places[i])

    # Reverse back and join
    result.reverse()
    return "".join(result)


def replace_numbers_with_thai(text: str):
    # Function to convert matched number to Thai text
    def convert_match(match: re.Match[str]):
        num_str = match.group(0).replace(',', '')

        # Skip if the string is empty or invalid after removing commas
        if not num_str or num_str == '.':
            return match.group(0)

        # Handle decimal numbers
        if '.' in num_str:
            parts = num_str.split('.')
            integer_part = parts[0]
            decimal_part = parts[1] if len(parts) > 1 else ''

            # If integer part is empty, treat as 0
            integer_value = int(integer_part) if integer_part else 0

            # If integer part is too long (>7 digits), read digit by digit
            if len(integer_part) > 7:
                result = number_to_thai_text(integer_value, digit_by_digit=True)
            else:
                result = number_to_thai_text(integer_value)

            # Add decimal part if it exists
            if decimal_part:
                result += "จุด " + " ".join(number_to_thai_text(int(d)) for d in decimal_part)
            return result

        # Handle integer numbers
        num = int(num_str)
        if len(num_str) > 7:  # If number exceeds 7 digits
            return number_to_thai_text(num, digit_by_digit=True)
        return number_to_thai_text(num)

    # Replace all numbers (with or without commas and decimals) in the text
    def process_text(text: str):
        # Split by spaces to process each word
        words = text.split()
        result = []

        for word in words:
            # Match only valid numeric strings (allowing commas and one decimal point)
            if re.match(r'^[\d,]+(\.\d+)?$', word):  # Valid number with optional decimal
                match_res = re.match(r'[\d,\.]+', word)
                if match_res is not None:
                    result.append(convert_match(match_res))
            else:
                # If word contains non-numeric characters, read numbers digit-by-digit
                if any(c.isdigit() for c in word):
                    processed = ""
                    num_chunk = ""
                    for char in word:
                        if char.isdigit():
                            num_chunk += char
                        else:
                            if num_chunk:
                                processed += " ".join(number_to_thai_text(int(d)) for d in num_chunk) + " "
                                num_chunk = ""
                            processed += char + " "
                    if num_chunk:  # Handle any remaining numbers
                        processed += " ".join(number_to_thai_text(int(d)) for d in num_chunk)
                    result.append(processed.strip())
                else:
                    result.append(word)

        return " ".join(result)

    return process_text(text)

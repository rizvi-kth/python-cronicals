from . import CSV_PATH
from datetime import datetime

def try_parse_float(string, fail=False):
    try:
        return float(string)
    except Exception:
        return fail


def get_first_digit_in_string(weight):
    # Process weight
    if weight:
        if isinstance(weight[0], str):
            weight[0] = weight[0].replace(',','.')
            ws = weight[0].split(' ')
            w_int = [w for w in ws if try_parse_float(w)] or False
            print(w_int[0])
            return float(w_int[0])
    return 0


def write_new_price_to_file(new_record):
    with open(CSV_PATH, mode='a') as file:
        # print('Writing to file :{:d}'.format(new_price))
        file.write('{:%Y-%m-%d %H:%M:%S}, {:s}\n'.format(datetime.now(), new_record))
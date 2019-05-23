
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

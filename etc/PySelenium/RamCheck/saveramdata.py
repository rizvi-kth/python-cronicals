import pandas as pd
from datetime import datetime


def write_new_price_to_file(new_price):
    with open("ram_price.csv", mode='a') as file:
        print('Writing to file :{:d}'.format(new_price))
        file.write('{:d}, {:%Y-%m-%d %H:%M:%S}\n'.format(new_price, datetime.now()))


def if_notify(new_price):
    df = None
    try:
        df = pd.read_csv('ram_price.csv', sep=",", parse_dates=True, header=None)  # index_col=0, # RamCheck/
    except:
        print('Could not read!')
        # But write
        write_new_price_to_file(new_price)
    else:
        print(df.head())

    # Check is it a different price than last time
    if df is not None and len(df.index) > 0:
        if df.iloc[-1:, 0:1].values[0][0] == new_price:
            print('No need to Notify! No need to write!')
            return False
        else:
            # And write
            write_new_price_to_file(new_price)
    return True


if __name__ == '__main__':
    print(if_notify(1204))

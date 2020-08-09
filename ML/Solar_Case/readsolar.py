import pandas as pd


def read_solar():
    sol_pdf = pd.read_csv('solar_data.csv', index_col=['Timestamp'], parse_dates=True)

    # Rename column
    sol_pdf.rename(columns={"Power (kW)": "Power",
                            "Month ": "Month",
                            "Global Horizontal Radiation (W/m²)": "GRad",
                            "Diffuse Horizontal Radiation (W/m²)":"DRad",
                            "Relative Humidity (%)":"Humid",
                            "Power (kW)":"Power",
                            "Temperature Celsius (°C)":"Temp",
                            "Wind Speed (m/s)":"Wind",
                            "Wind Direction (Degrees)":"WindD",
                            "Daily Rainfall (mm)":"Rain",
                            "Performance Ratio (%)":"Performance",
                            "Current Phase Average (A)":"Current"
                           }, inplace=True)

    # Format time column
    # sol_pdf['Timest'] = pd.to_datetime(sol_pdf['Timestamp'])
    # print(sol_pdf.dtypes)

    # Set index to query on time seris data
    # sol_id_pdf = sol_pdf.set_index('Timest')
    # print(sol_id_pdf.index)

    return sol_pdf # sol_id_pdf


if __name__ == '__main__':
    read_solar()




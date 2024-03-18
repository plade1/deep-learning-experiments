import os
import polars as pl
from typing import Literal
from pathlib import Path

TEST_DATA_FOLDER = Path(__file__).parent.joinpath("test")


def load_test_cycler_data(file_name):
    """
    load the test cycler data

    "Test_1_ArbinCyclerData_Cyc_360V250V1C_45C_00014_c3.parquet"
    need to merge with the cycle protocol
    """
    df1 = pl.read_parquet(
        TEST_DATA_FOLDER.joinpath(
            "Test_1_ArbinCyclerData_Cyc_360V250V1C_45C_00014_c3.parquet"
        ).absolute()
    )
    df2 = pl.read_parquet(
        TEST_DATA_FOLDER.joinpath(
            "Test_2_ArbinCyclerData_Cyc_360V250V1C_45C_00014_c3.parquet"
        ).absolute()
    )
    cycle_protocol = (
        pl.read_parquet(
            TEST_DATA_FOLDER.joinpath(
                "cyc-lfp-ummlp_45c+LFP_UMMLP_5N6P_EL09-06.parquet"
            ).absolute()
        )
        .drop("timestamp")
        .with_columns(pl.col("Step_Index").cast(pl.Int64))
    )
    df1_new = df1.join(
        cycle_protocol, left_on="Step_Index", right_on="Step_Index"
    ).with_columns(pl.lit(0.0).alias("coulombic_efficiency"))
    df1_new = df1_new.select(sorted(df1_new.columns))
    df2 = df2.select(sorted(df2.columns))

    df2 = df2.with_columns(
        [
            pl.col("Channel").cast(pl.Int64),
            pl.col("Charge_Energy").str.strip(" ").cast(pl.Float64),
            pl.col("Data_Time").cast(pl.Int64),
        ]
    )

    raw_df = (
        pl.concat([df1_new, df2]).unique(subset=["Data_Point"]).sort(by=["Data_Point"])
    ).rename(
        {
            "Test_Time": "Test_Time(s)",
            "Step_Time": "Step_Time(s)",
            "Voltage": "Voltage(V)",
            "Current": "Current(A)",
            "Charge_Energy": "Charge_Energy(Wh)",
            "Discharge_Energy": "Discharge_Energy(Wh)",
            "Discharge_Capacity": "Discharge_Capacity(Ah)",
            "Charge_Capacity": "Charge_Capacity(Ah)",
        }
    )

    return raw_df


import tensorflow as tf

# Define the model
model = tf.keras.models.Sequential()

# Add LSTM layer with variable-length input
model.add(tf.keras.layers.LSTM(50, input_shape=(None, 2)))

# Add a Dense layer
model.add(tf.keras.layers.Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

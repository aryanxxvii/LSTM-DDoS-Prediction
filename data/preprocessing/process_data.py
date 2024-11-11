from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, concat_ws
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StandardScaler, VectorAssembler, StringIndexer
import numpy as np
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_ROOT = "./data/data"
OUTPUT_PATH = "./data/test"
FILES = [
    "dns.csv", "ldap.csv", "mssql.csv", "netBIOS.csv", "ntp.csv",
    "snmp.csv", "ssdp.csv", "syn.csv", "tftp.csv", "udp.csv", "udplag.csv"
]

COLUMNS_TO_DROP_UNIQUE = ['Flow ID', ' Fwd Header Length.1']
COLUMNS_TO_DROP_MANUAL = [
    'Fwd PSH Flags', ' RST Flag Count', ' Total Fwd Packets',
    ' Total Backward Packets', ' Total Length of Bwd Packets',
    ' Fwd Header Length', 'Subflow Fwd Packets', ' Subflow Bwd Packets',
    ' Subflow Bwd Bytes', ' Subflow Fwd Bytes', ' Fwd Packet Length Mean',
    ' Flow Duration', ' Flow IAT Max', 'Fwd IAT Total', ' Idle Max',
    'Bwd Packet Length Max', ' Fwd Packet Length Max', ' Fwd Packet Length Std',
    ' Bwd Packet Length Mean', ' Bwd Packet Length Std', ' Flow IAT Mean',
    ' Flow IAT Std', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max',
    'Idle Mean', ' Fwd IAT Min', ' Bwd IAT Min', ' Bwd IAT Std',
    ' Bwd IAT Max', ' Max Packet Length', ' Packet Length Std',
    ' Packet Length Variance', ' Average Packet Size', 'Active Mean',
    ' Active Max', 'Flow Bytes/s', ' Flow Packets/s'
]

def create_spark_session():
    """Create and return a Spark session"""
    return SparkSession.builder \
        .appName("DDoS Data Processing") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

def load_and_combine_data(spark, files):
    """Load and combine all CSV files"""
    logger.info("Loading and combining CSV files...")
    dfs = []
    for file in files:
        path = os.path.join(DATA_ROOT, file)
        if os.path.exists(path):
            df = spark.read.csv(path, header=True, inferSchema=True)
            dfs.append(df)
        else:
            logger.warning(f"File not found: {path}")
    
    return dfs[0].unionAll(*dfs[1:])

def preprocess_data(df):
    """Apply all preprocessing steps"""
    logger.info("Starting preprocessing...")
    
    df = df.withColumn(" Timestamp", to_timestamp(col(" Timestamp"))) \
           .orderBy(" Timestamp")

    for column in df.columns:
        if df.select(column).distinct().count() == 1:
            df = df.drop(column)

    columns_to_drop = COLUMNS_TO_DROP_UNIQUE + COLUMNS_TO_DROP_MANUAL
    df = df.drop(*columns_to_drop)

    ip_indexer_src = StringIndexer(inputCol=" Source IP", outputCol="source_ip_encoded")
    ip_indexer_dst = StringIndexer(inputCol=" Destination IP", outputCol="destination_ip_encoded")
    
    df = ip_indexer_src.fit(df).transform(df)
    df = ip_indexer_dst.fit(df).transform(df)
    
    df = df.drop(" Source IP", " Destination IP") \
           .withColumnRenamed("source_ip_encoded", " Source IP") \
           .withColumnRenamed("destination_ip_encoded", " Destination IP")

    label_indexer = StringIndexer(inputCol=" Label", outputCol="label_encoded")
    df = label_indexer.fit(df).transform(df)
    df = df.drop(" Label").withColumnRenamed("label_encoded", " Label")

    numeric_cols = []
    for column in df.columns:
        if column != " Timestamp":
            df = df.withColumn(column, col(column).cast(DoubleType()))
            if column != " Label":
                numeric_cols.append(column)

    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
    df = assembler.transform(df)
    
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)

    return df

def save_to_numpy(df):
    """Convert to numpy array and save to pickle file"""
    logger.info("Converting to numpy array and saving...")
    
    pandas_df = df.toPandas()
    pandas_df.set_index(" Timestamp", inplace=True)
    
    X = np.array(pandas_df.drop(" Label", axis=1))
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    with open(os.path.join(OUTPUT_PATH, "testX.pkl"), "wb") as f:
        pickle.dump(X, f)
    
    logger.info(f"Data saved to {OUTPUT_PATH}")
    return X

def main():
    try:
        spark = create_spark_session()
        
        df = load_and_combine_data(spark, FILES)
        
        df_processed = preprocess_data(df)
        
        X, y = save_to_numpy(df_processed)
        
        logger.info(f"Processing complete. Features shape: {X.shape}, Labels shape: {y.shape}")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
    finally:
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    main()
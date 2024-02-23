import datetime as dt
from datetime import timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
# from airflow.operators.python_operator import PyhonOperator
import pandas as pd
import great_expectations as ge
import os
import matplotlib.pyplot as plt 
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
# from sklearn.externals import joblib
import joblib



def transform_data():
    df=pd.read_csv("/home/noname00/Documents/DE_SCHEADULING/SERVER_MAIN/BankCustomerChurnPrediction.csv")
    df.loc[df.gender=="Female","gender"]="F"
    df.loc[df.gender=="Male","gender"]="M"
    df.to_json("/home/noname00/Documents/DE_SCHEADULING/SERVER_MAIN/TapTarget/customer.json", orient="records")

def validate_data():
    isValid=True
    df=pd.read_json("/home/noname00/Documents/DE_SCHEADULING/SERVER_MAIN/TapTarget/customer.json")
    ge_cs=ge.from_pandas(df)
    if(ge_cs.expect_column_values_to_be_unique("customer_id")[0]!="success"):
        isValid=False
    # cek apakah ada nilai null
    for column in df.columns:
        if(ge_cs.expect_column_values_to_not_be_null(column)[0]!="success"):
            isValid=False
    if not isValid :
        raise ValueError("Data Tidak Valid !")
    # os.system("chmod 777 /home/noname00/Documents/DE_SCHEADULING/SERVER_MAIN/TapTarget/customer.json")
def visualize():
    color = '#eab889'
    df=pd.read_json("/home/noname00/Documents/DE_SCHEADULING/SERVER_MAIN/TapTarget/customer.json")
    df.hist(bins=15,figsize=(25,15),color=color)
    plt.rcParams['font.size'] = 18
    plt.savefig("/home/noname00/Documents/DE_SCHEADULING/SERVER_MAIN/Visualisasi/distribusi_data.png")
def modeling():
    df=pd.read_json("/home/noname00/Documents/DE_SCHEADULING/SERVER_MAIN/TapTarget/customer.json")
    # Melakukan proses encoding pada fitur country dan gender
    df['country'] = LabelEncoder().fit_transform(df['country'])
    df['gender'] = LabelEncoder().fit_transform(df['gender'])
    # Fitur customer_id tidak digunakan karena tidak ada korelasi dengan kelas churn
    df=df.drop('customer_id',axis=1)
    # x sebagai fitur dan y sebagai kelas
    x= df.copy()
    x.drop('churn',axis = 1,inplace = True)
    y = df['churn']
    # melakukan populasi data dengan SMOTE karena data yang ada imbalance
    smk = SMOTETomek(random_state=123)
    x_res , y_res = smk.fit_resample(x,y)
    # membuat data training dan test serta melakukan normalisasi
    X_train, X_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.2,random_state=0)
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    """# Modeling"""

    # Membuat model menggunakan Random Forest Classifier dengan banyaknya  cpu
    import multiprocessing
    cpu=multiprocessing.cpu_count()
    # Training data akan dilakukan dengan parallel processing menggunakan n_jobs(jumlah cpu) = 12
    model = RandomForestClassifier(n_estimators=500, n_jobs=cpu)

    # membuat model
    model.fit(X_train, y_train)

    # Validasi model dengan Stratified K-Fold
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
    n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=cpu)
    print(n_scores)
    print("Dengan rata-rata : ",(sum(n_scores)/len(n_scores))*100," %")


    predict=model.predict(X_test)
    print("Akurasi dari prediksi : ",(sum(predict==y_test)/len(y_test))*100," %")
    filename = "/home/noname00/Documents/DE_SCHEADULING/SERVER_MAIN/Model/Completed_model.joblib"
    with open('/home/noname00/Documents/DE_SCHEADULING/SERVER_MAIN/Model/model_details.txt', 'w') as f:
        f.write("".join(['rata-rata kfold : ',str((sum(n_scores)/len(n_scores))*100),"\n","Akurasi :",str((sum(predict==y_test)/len(y_test))*100)]))
    joblib.dump(model, filename)




default_args={
    "owner" : "Kelompok6",
    "start_date" : datetime.utcnow(),
}
with DAG("DAG_kelompokEnam",default_args=default_args,schedule_interval=timedelta(minutes=1),catchup=False) as dag:
    extract=BashOperator(task_id="extract",bash_command="mv /home/noname00/Documents/DE_SCHEADULING/SERVER_A/BankCustomerChurnPrediction.csv /home/noname00/Documents/DE_SCHEADULING/SERVER_MAIN")
    transforms=PythonOperator(task_id="transform",python_callable=transform_data)
    validates=PythonOperator(task_id="validate",python_callable=validate_data)
    TapTarget=BashOperator(task_id="load",bash_command="python3 /home/noname00/Documents/DE_SCHEADULING/SERVER_MAIN/TapTarget/tap_emp_api.py | /home/noname00/Documents/DE_SCHEADULING/SERVER_MAIN/TapTarget/Singer.io_postgres_target/bin/target-postgres -c /home/noname00/Documents/DE_SCHEADULING/SERVER_MAIN/TapTarget/Singer.io_postgres_target/bin/config.json")
    visual=PythonOperator(task_id="visual",python_callable=visualize)
    models=PythonOperator(task_id="model",python_callable=modeling)

extract>>transforms>>validates>>TapTarget>>visual>>models
